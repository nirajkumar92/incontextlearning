"""Check retained research artifacts, citations, budgets and source provenance.

This checks consistency, not whether the scientific hypothesis wins benchmarks.
Native LaTeX compilation and executable tests are recorded separately.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re


def audit(root):
    def read(name):
        return json.loads((root / name).read_text())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    spec = read('research/specs/prior_bank_spec.json')
    finance = read('research/specs/finance_recipe_v3.json')
    assert sum(spec['compute']['phase_gpu_hours'].values()) == 50000
    screen = spec['compute']['screening_detail']
    assert sum(row['total'] for row in screen['allocation']) == screen['total'] == 15000
    for row in screen['allocation']:
        assert row['arms'] * row['seeds'] * row['hours_per_arm_seed'] == row['total'], row['id']
    scaling = spec['compute']['scaling_runs']
    assert sum(scaling[k]['total'] for k in ['short_runs', 'bridges', 'resolution']) == scaling['total'] == 10000
    projected = read('research/results/compute_plan_50000.json')['budget']['phase_gpu_hours']
    expected = dict(spec['compute']['phase_gpu_hours'])
    expected['prior_screening'] = expected.pop('prior_architecture_optimizer_screening')
    assert projected == expected, 'compute calculator and research allocation disagree'
    assert spec['compute']['total_gpus'] == 64
    candidate = spec['main_candidate']
    standard = read(candidate['standard_config'])
    specialist = read(candidate['fraud_config'])
    assert standard['model'] == specialist['model'] == candidate['model']
    assert standard['model_options'] == specialist['model_options'] == candidate['model_options']
    assert standard['standard_prior'] == candidate['standard_prior']
    assert standard['finance_share'] == 0 and specialist['finance_share'] == 1
    assert specialist['finance_task'] == candidate['fraud_prior']['task'] == 'binary'
    assert specialist['finance_overrides']['prevalence'] == 1e-4
    assert sum(candidate['budget_gpu_hours'].values()) == 12000
    volume = read(candidate['volume_plan'])
    assert volume['project_gpu_hour_ceiling'] == 50000
    assert volume['standard']['provisional_target_accepted_macroepisodes'] == candidate['standard_accepted_episode_target']
    assert volume['finance']['provisional_target_accepted_macroepisodes'] == candidate['fraud_macroepisode_target']
    for phase in ('standard', 'finance'):
        assert volume[phase]['global_batch'] == candidate['global_batch']
    pin_root = root / 'third_party' / 'tabicl_reference'
    pin = json.loads((pin_root / 'manifest.json').read_text())
    assert pin['revision'] == candidate['reference_revision']
    for filename, entry in pin['files'].items():
        assert digest(pin_root / filename) == entry['sha256'], filename
    assert spec['profile_mixture']['target_fraud_rate'] == 1e-4
    assert finance['finite_world']['prevalence']['operational_target_prevalence'] == 1e-4
    for path in spec['selected_files'].values():
        assert (root / 'research' / 'specs' / path).is_file(), path
    joint = finance['joint_profile']
    for task in ['binary', 'multiclass', 'regression']:
        marginal = joint['standard_probability'] * joint['standard_marginal_head_probabilities'][task] + joint['finance_probability'] * joint['finance_head_probabilities'][task]
        assert abs(marginal - 1/3) < 1e-12
    document = root / 'prior_bank_assessment.tex'
    tex = document.read_text()
    bib = re.findall(r'\\bibitem\{([^}]+)\}', tex)
    citations = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', tex) for key in group.split(',')}
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    refs = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex))
    assert len(bib) == len(set(bib)) and not citations.difference(bib)
    assert len(labels) == len(set(labels)) and not refs.difference(labels)
    stack = []
    for command, environment in re.findall(r'\\(begin|end)\{([^}]+)\}', tex):
        if command == 'begin':
            stack.append(environment)
        else:
            assert stack and stack.pop() == environment, environment
    assert not stack
    sources = read('papers/source_manifest.json')['papers']
    for source in sources:
        path = root / source['path']
        assert path.is_file() and digest(path) == source['sha256'], source['path']
    with (root / 'research/specs/ablation_plan.csv').open() as file:
        ablations = list(csv.DictReader(file))
    return {'source': document.name, 'sha256': digest(document), 'resolved_citation_keys': len(citations),
            'bibliography_entries': len(bib), 'ablation_rows': len(ablations),
            'retained_source_hashes_verified': len(sources), 'selected_specs_and_budget': 'passed',
            'screening_and_bridge_budget_arithmetic': 'passed',
            'candidate_config_volume_consistency': 'passed',
            'pinned_reference_files_verified': len(pin['files']),
            'internal_references_and_environment_nesting': 'passed', 'target_fraud_rate': 1e-4,
            'native_compile': 'not run by this script', 'competitive_performance': 'not evaluated'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
