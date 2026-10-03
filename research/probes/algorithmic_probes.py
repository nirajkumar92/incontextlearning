"""Analytic learnability and Rao-Blackwell gradient probes, not model benchmarks."""
import argparse
import json
from pathlib import Path
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def run(seed=20261003, worlds=30000, replicates=32):
    rng = np.random.default_rng(seed)
    n_support, n_query = 32, 8
    theta = rng.choice([-2., 2.], size=worlds)
    xs = rng.normal(size=(worlds, n_support))
    ys = rng.binomial(1, sigmoid(theta[:, None] * xs))
    context_stat = np.mean(xs * (2 * ys - 1), axis=1)
    xq = rng.normal(size=(worlds, n_query))
    features = np.stack([context_stat[:, None] * xq, np.ones_like(xq)], axis=-1)
    student_probability = sigmoid(features @ np.array([2.0, 0.1]))
    true_probability = sigmoid(theta[:, None] * xq)
    # Oracle probability is loss-only; the student's features use support and xq.
    soft_gradient = np.mean((student_probability - true_probability)[..., None] * features, axis=1)
    expected_noise_trace = np.mean(np.sum(
        true_probability * (1 - true_probability) * np.sum(features**2, axis=-1), axis=1) / n_query**2)
    reductions, mean_errors = [], []
    for _ in range(replicates):
        yq = rng.binomial(1, true_probability)
        hard_gradient = np.mean((student_probability - yq)[..., None] * features, axis=1)
        reductions.append(float(np.mean(np.sum((hard_gradient-soft_gradient)**2,axis=1))))
        mean_errors.append(np.mean(hard_gradient-soft_gradient,axis=0))
    reductions=np.array(reductions)
    se=float(reductions.std(ddof=1)/np.sqrt(replicates))
    assert abs(reductions.mean()-expected_noise_trace) < 6*se
    tree=[]
    for n in (32,128,512):
        for depth in (3,5,8,10):
            leaves=2**depth
            p_empty=(1-1/leaves)**n
            tree.append(dict(support=n,depth=depth,leaves=leaves,expected_support_per_leaf=n/leaves,
                             probability_query_leaf_unseen=p_empty,
                             binary_bayes_accuracy_upper_bound=1-.5*p_empty))
    ols=[]
    for dimension in (4,16,32,48,60):
        n=64
        ols.append(dict(support=n,effective_dimension=dimension,
                        test_mse_with_unit_noise=1+dimension/(n-dimension-1)))
    return {
        'status':'Executed analytic/simulation diagnostics, not a trained PFN or real-data result',
        'tree_assumptions':'Known equal-probability leaves, independent Bernoulli(.5) leaf labels, noiseless support; unknown tree splits can only make prediction harder.',
        'tree_occupancy':tree,
        'ols_assumptions':'Correctly specified no-intercept Gaussian isotropic linear model, independent unit-variance Gaussian noise, OLS, n>d+1; illustrative estimation cost, not optimal Bayes risk.',
        'ols_effective_dimension':ols,
        'rao_blackwell':{
            'worlds':worlds,'label_replicates_per_fixed_world':replicates,'seed':seed,
            'support_per_world':n_support,'query_per_world':n_query,
            'student':'Fixed two-parameter context-statistic logistic predictor; no optimization performed',
            'soft_gradient_variance_trace_across_worlds':float(soft_gradient.var(axis=0,ddof=1).sum()),
            'analytic_removed_label_noise_gradient_mse':float(expected_noise_trace),
            'observed_removed_label_noise_gradient_mse':float(reductions.mean()),
            'monte_carlo_standard_error':se,
            'mean_hard_minus_soft_gradient':np.mean(mean_errors,axis=0).tolist(),
            'six_standard_error_identity_check':'passed',
            'limitation':'Does not prove faster training, downstream gains, or that a plug-in forward mechanism is valid for anti-causal SCM targets.',
        },
    }

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(run(),indent=2)+'\n')
    print(a.output)
