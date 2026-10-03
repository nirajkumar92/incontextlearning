"""Task-trained tree comparisons on explicit train/validation/test snapshots.

Optional dependencies are installed with .[trees]. Tuning only sees validation
outcomes. Natural-prevalence test predictions are evaluated once after selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .data import load_table,validate_pair,assert_time_contract,assert_validation_time_contract
from .metrics import binary_metrics,multiclass_metrics,regression_metrics,select_threshold
from .runtime import atomic_json


def _frames(support, others, method):
    import pandas as pd
    cat=support['categorical']
    def frame(X):
        return pd.DataFrame(X,columns=[f'x{i}' for i in range(X.shape[1])])
    frames=[frame(support['X'])]+[frame(t['X']) for t in others]
    for j in np.flatnonzero(cat):
        name=f'x{j}'
        if method=='catboost':
            for f in frames:
                f[name]=f[name].map(lambda v:'__MISSING__' if not np.isfinite(v) else 'category:'+float(v).hex())
        else:
            levels=np.unique(support['X'][np.isfinite(support['X'][:,j]),j])
            for f in frames:
                f[name]=pd.Categorical(f[name],categories=levels)
    return frames


def _parameters(method,rng,trial,rounds,seed):
    lu=lambda a,b:float(np.exp(rng.uniform(np.log(a),np.log(b))))
    common={'learning_rate':lu(.01,.2),'n_estimators':rounds,'random_state':seed,'n_jobs':1}
    if method=='xgboost':
        if trial==0:
            return {'n_estimators':rounds,'tree_method':'hist','enable_categorical':True,'n_jobs':1,'random_state':seed,'early_stopping_rounds':100}
        return dict(common,tree_method='hist',enable_categorical=True,max_depth=int(rng.choice([3,5,7,9])),
                    min_child_weight=lu(1,100),subsample=float(rng.uniform(.6,1)),colsample_bytree=float(rng.uniform(.6,1)),reg_lambda=lu(.1,100),early_stopping_rounds=100)
    if method=='lightgbm':
        if trial==0:
            return {'n_estimators':rounds,'n_jobs':1,'random_state':seed,'verbosity':-1}
        return dict(common,num_leaves=int(rng.choice([15,31,63,127])),min_child_samples=round(lu(20,2000)),
                    subsample=float(rng.uniform(.6,1)),subsample_freq=1,colsample_bytree=float(rng.uniform(.6,1)),reg_lambda=lu(.1,100),verbosity=-1)
    if method=='catboost':
        result={'iterations':rounds,'random_seed':seed,'thread_count':1,'verbose':False,'allow_writing_files':False}
        if trial:
            result.update(depth=int(rng.choice([4,6,8,10])),learning_rate=lu(.01,.2),l2_leaf_reg=lu(1,100))
        return result
    raise ValueError('Unknown tree library')


class _MappedClassifier:
    """Expose original labels while a library sees consecutive observed IDs."""
    def __init__(self, estimator, observed_classes, validation_seen):
        self.estimator = estimator
        self.classes_ = np.asarray(observed_classes, dtype=int)
        self.validation_seen = np.asarray(validation_seen, dtype=bool)

    def predict_proba(self, X):
        if self.estimator is None:
            return np.ones((len(X), 1))
        probabilities = np.asarray(self.estimator.predict_proba(X))
        # Do not assume a library's probability-column order implicitly.
        internal = np.asarray(self.estimator.classes_, dtype=int)
        result = np.zeros((len(X), len(self.classes_)))
        result[:, internal] = probabilities
        return result


def _take_rows(X, mask):
    return X.iloc[np.flatnonzero(mask)] if hasattr(X, 'iloc') else X[mask]


def fit_candidate(method,task,parameters,X,y,V,yv,categorical,sample_weight=None):
    classification=task!='regression'
    if method not in ('xgboost', 'lightgbm', 'catboost'):
        raise ValueError('Unknown tree library')
    y,yv=np.asarray(y),np.asarray(yv)
    if not len(y):
        raise ValueError('A tree baseline requires nonempty training support')
    parameters=dict(parameters)
    validation_seen=np.ones(len(yv),dtype=bool)
    if classification:
        if (not np.isfinite(y).all() or not np.isfinite(yv).all() or
                not np.equal(y,np.floor(y)).all() or not np.equal(yv,np.floor(yv)).all()):
            raise ValueError('Classification labels must be finite integers')
        observed=np.unique(y).astype(int)
        validation_seen=np.isin(yv,observed)
        if len(observed)==1:
            return _MappedClassifier(None,observed,validation_seen)
        y=np.searchsorted(observed,y)
        V=_take_rows(V,validation_seen)
        yv=np.searchsorted(observed,yv[validation_seen])
    use_validation=bool(len(yv))
    if method=='xgboost':
        import xgboost as xgb
        if not use_validation:
            parameters.pop('early_stopping_rounds',None)
        cls=xgb.XGBClassifier if classification else xgb.XGBRegressor
        model=cls(**parameters)
        model.fit(X,y,eval_set=[(V,yv)] if use_validation else None,sample_weight=sample_weight,verbose=False)
    elif method=='lightgbm':
        import lightgbm as lgb
        cls=lgb.LGBMClassifier if classification else lgb.LGBMRegressor
        model=cls(**parameters)
        model.fit(X,y,eval_set=[(V,yv)] if use_validation else None,sample_weight=sample_weight,
                  callbacks=[lgb.early_stopping(100,verbose=False)] if use_validation else [])
    else:
        from catboost import CatBoostClassifier,CatBoostRegressor
        cls=CatBoostClassifier if classification else CatBoostRegressor
        model=cls(**parameters)
        model.fit(X,y,eval_set=(V,yv) if use_validation else None,
                  cat_features=list(np.flatnonzero(categorical)),sample_weight=sample_weight,
                  early_stopping_rounds=100 if use_validation else None)
    return _MappedClassifier(model,observed,validation_seen) if classification else model


def _predict(model,X,task,classes):
    if task=='regression':
        return model.predict(X)
    result=np.zeros((len(X),classes))
    prediction=model.predict_proba(X)
    result[:,np.asarray(model.classes_,dtype=int)]=prediction
    return result


def _validate_snapshots(S,V,T,task,classes,cutoff):
    validate_pair(S,V);validate_pair(S,T);validate_pair(V,T)
    if any(len(table['X'])==0 for table in (S,V,T)):
        raise ValueError('Train, validation and test snapshots must each be nonempty')
    if task!='regression':
        if classes<2 or (task=='binary' and classes!=2):
            raise ValueError('Binary needs exactly two classes; multiclass needs at least two')
        for name,table in (('train',S),('validation',V),('test',T)):
            if not np.isin(table['y'],np.arange(classes)).all():
                raise ValueError(f'Invalid {name} class labels for declared universe')
    if cutoff is not None:
        assert_time_contract(S,V,cutoff);assert_time_contract(S,T,cutoff)
        assert_validation_time_contract(V,T)


def _class_coverage(S,V,T,classes):
    observed=np.unique(S['y']).astype(int)
    return {'declared_classes':classes,'observed_train_classes':observed.tolist(),
            'unseen_train_classes':np.setdiff1d(np.arange(classes),observed).tolist(),
            'validation_rows':len(V['y']),
            'validation_unseen_class_rows':int((~np.isin(V['y'],observed)).sum()),
            'test_rows':len(T['y']),'test_unseen_class_rows':int((~np.isin(T['y'],observed)).sum()),
            'unseen_class_policy':'Zero predicted mass; retain all rows in selection and final scoring. Log loss clips at 1e-15.',
            'early_stopping_policy':'Observed-class validation rows only; disabled when none overlap training.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['xgboost','lightgbm','catboost'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--test',required=True)
    p.add_argument('--task',choices=['binary','multiclass','regression'],required=True)
    p.add_argument('--classes',type=int,default=2);p.add_argument('--trials',type=int,default=32)
    p.add_argument('--rounds',type=int,default=5000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seconds',type=float,default=3600);p.add_argument('--cutoff',type=float)
    p.add_argument('--minimum-precision',type=float,default=.8);p.add_argument('--review-budget',type=int)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.trials<1 or args.rounds<1 or args.seconds<=0:
        raise ValueError('Positive trial, round and wall-time budgets required')
    S,V,T=load_table(args.train),load_table(args.validation),load_table(args.test)
    _validate_snapshots(S,V,T,args.task,args.classes,args.cutoff)
    X,Xv,Xt=_frames(S,[V,T],args.method)
    rng=np.random.default_rng(args.seed);start=time.perf_counter();records=[];best=None;best_score=float('inf')
    if args.task!='regression' and len(np.unique(S['y']))<2:
        # A trained tree cannot split a constant target. Preserve this case and
        # disclose a constant predictor rather than deleting the test window.
        model=fit_candidate(args.method,args.task,{},X,S['y'],Xv,V['y'],S['categorical'])
        pv=_predict(model,Xv,args.task,args.classes)
        pt=_predict(model,Xt,args.task,args.classes)
        records=[{'constant_target_fallback':True}]
    else:
        for trial in range(args.trials):
            if trial and time.perf_counter()-start>=args.seconds:
                break
            params=_parameters(args.method,rng,trial,args.rounds,args.seed+trial)
            weights=None;positive_multiplier=1.
            if args.task=='binary' and trial%2==1:
                m=int(np.sum(S['y']==1));n=int(np.sum(S['y']==0))
                cap=min(1000,n/max(1,m))
                if cap>1:
                    positive_multiplier=float(np.exp(rng.uniform(0,np.log(cap))))
                    weights=np.where(S['y']==1,positive_multiplier,1.)
            model=fit_candidate(args.method,args.task,params,X,S['y'],Xv,V['y'],S['categorical'],weights)
            pv=_predict(model,Xv,args.task,args.classes)
            if args.task=='binary':
                metric=binary_metrics(V['y'],pv[:,1]);score=-metric['average_precision'] if np.sum(V['y']) else metric['log_loss']
            elif args.task=='multiclass':
                score=multiclass_metrics(V['y'],pv)['log_loss']
            else:
                score=regression_metrics(V['y'],pv)['rmse']
            records.append({'trial':trial,'parameters':params,'positive_multiplier':positive_multiplier,'validation_objective':score,
                            'early_stopping_validation_rows':int(model.validation_seen.sum()) if args.task!='regression' else len(V['y'])})
            if score<best_score:
                best,best_score=model,score
        pv=_predict(best,Xv,args.task,args.classes);pt=_predict(best,Xt,args.task,args.classes)
    if args.task=='binary':
        threshold=select_threshold(V['y'],pv[:,1],args.minimum_precision)
        metrics=binary_metrics(T['y'],pt[:,1],threshold,args.review_budget)
    elif args.task=='multiclass':
        metrics=multiclass_metrics(T['y'],pt)
    else:
        metrics=regression_metrics(T['y'],pt)
    atomic_json(args.output,{'method':args.method,'track':'full_history_task_trained','trials':records,'test':metrics,
             'class_coverage':_class_coverage(S,V,T,args.classes) if args.task!='regression' else None,
             'temporal_contract_checked':args.cutoff is not None,
             'seconds':time.perf_counter()-start,'requested_seconds':args.seconds,
             'wall_cap_note':'Checked between trials; the last trial can exceed the wall-time budget.',
             'validation_selection':'binary AP (log loss if no positives), multiclass log loss, regression RMSE',
             'probability_calibration':'none; class-weighted scores are not asserted calibrated'})


if __name__=='__main__':
    main()
