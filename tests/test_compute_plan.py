"""Meaningful unit/accounting checks; no accelerator performance tests."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'probes'))
from compute_plan import project_budget, dense_row_train_flops, profile_to_plan


class ComputePlanTests(unittest.TestCase):
    def test_project_conservation_and_gpu_wall_time(self):
        b = project_budget()
        self.assertAlmostEqual(sum(b['phase_gpu_hours'].values()), 50_000)
        self.assertAlmostEqual(b['full_cluster_days'], 50_000 / 64 / 24)
        self.assertEqual(b['phase_gpu_hours']['final_pretraining'], 12_000)

    def test_gpu_hours_multiply_allocated_devices_not_wall_hours(self):
        r = dict(config_id='unit-test', params=1000, allocated_gpus=8,
                 wall_seconds=10, accepted_episodes=4, rows=40, cells=120,
                 query_targets=8, core_tokens=40,
                 core_token_definition='one per row')
        plan = profile_to_plan(r, 1)
        self.assertEqual(plan['gpu_seconds_per_accepted_episode'], 20)
        self.assertEqual(plan['projected_accepted_episodes'], 180)
        self.assertEqual(plan['projected_processed_counters']['query_targets'], 360)
        self.assertAlmostEqual(plan['projected_core_tokens_per_parameter'], 1.8)

    def test_attention_cost_is_quadratic_not_just_six_nd(self):
        n,l,d,s = 100,2,4,8
        c1=dense_row_train_flops(n,l,d,s)
        c2=dense_row_train_flops(n,l,d,2*s)
        self.assertGreater(c2,2*c1)
        self.assertEqual(c2-2*c1,24*l*d*s*s)
        self.assertEqual(dense_row_train_flops(n,l,d,s,3),3*c1)

    def test_invalid_and_undefined_measurements_are_rejected(self):
        for bad in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): project_budget(bad)
        with self.assertRaises(ValueError): project_budget(gpus=1.5)
        r=dict(config_id='invalid',params=100,allocated_gpus=1,wall_seconds=1,
               accepted_episodes=1,rows=2,cells=4,query_targets=1,core_tokens=2,
               core_token_definition='')
        with self.assertRaises(ValueError): profile_to_plan(r,1)
        r['core_token_definition']='row'; r['accepted_episodes']=0
        with self.assertRaises(ValueError): profile_to_plan(r,1)

if __name__ == '__main__': unittest.main()
