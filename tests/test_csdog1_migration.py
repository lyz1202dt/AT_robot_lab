"""CPU integration checks for the full Cyberstar task (no Isaac Sim imports)."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/csdog1/cyberstar"
spec = importlib.util.spec_from_file_location("csdog1_migration",PACKAGE / "__init__.py",submodule_search_locations=[str(PACKAGE)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from csdog1_migration.runtime import CyberstarRuntime


def load_checkpoint_helper():
    path = ROOT / "source/robot_lab/robot_lab/him/source_checkpoint.py"
    spec = importlib.util.spec_from_file_location("csdog1_checkpoint_helper",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.migrate_source_state


class MigrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.runtime = CyberstarRuntime(16,"cpu")

    def test_history_reset_and_critic_layout(self):
        r = self.runtime
        r.add_noise = False
        r.compute_observations()
        self.assertEqual(r.obs_buf.shape,(16,282))
        self.assertEqual(r.privileged_obs_buf.shape,(16,491))
        torch.testing.assert_close(r.obs_buf.reshape(16,6,47),r.clean_actor_frame[:,None].expand(-1,6,-1))
        r.base_lin_vel[:] = torch.tensor([.5,-.2,.1])
        r.commands[:,0] = .75
        r.compute_observations()
        torch.testing.assert_close(r.privileged_obs_buf[:,282:285],torch.tensor([1.,-.4,.2]).expand(16,-1))
        torch.testing.assert_close(r.obs_buf[:,-39],torch.full((16,),1.5))
        r.reset(torch.tensor([2,4]))
        r.compute_observations()
        for env_id in (2,4):
            frames = r.obs_buf[env_id].reshape(6,47)
            torch.testing.assert_close(frames,frames[-1].expand_as(frames))

    def test_policy_step_delay_and_recovery_clipping(self):
        r = self.runtime
        r.control_delay[::2] = 1
        r.process_actions(torch.full((16,12),2.))
        torch.testing.assert_close(r.actions[::2],torch.zeros(8,12))
        torch.testing.assert_close(r.actions[1::2],torch.full((8,12),2.))
        r.process_actions(torch.full((16,12),3.))
        torch.testing.assert_close(r.actions[::2],torch.full((8,12),2.))
        r.ppo_iteration = 1800
        r.capture_recovery_active[1] = True
        r.process_actions(torch.full((16,12),8.))
        torch.testing.assert_close(r.actions[1],torch.full((12,),1.5))
        self.assertEqual(r.actions[3,0],8.)

    def test_observation_clip_preserves_internal_history(self):
        r = self.runtime
        r.add_noise = False
        r.dof_vel[:] = 10000
        r.compute_observations()
        self.assertEqual(r.obs_buf.max(), 100)
        self.assertEqual(r.privileged_obs_buf.max(), 100)
        self.assertGreater(r.clean_observation_history.max(), 100)

    def test_stop_requires_adjustment_and_supported_completion(self):
        r = self.runtime
        r.command_steps_remaining[:] = 10000
        r.commands[:,0] = .5
        r.command_targets[:,0] = .5
        r.contact_forces[:,:4,2] = 35
        r._post_physics_step_callback()
        r.commands[:] = 0
        r.command_targets[:] = 0
        r._post_physics_step_callback()
        self.assertTrue(r.stop_adjustment_active.all())
        # Being motionless with all feet down cannot bypass the required lift/landing window.
        for _ in range(15):
            r._post_physics_step_callback()
        self.assertTrue(r.stop_adjustment_active.all())
        self.assertTrue((r.pose_recovery_blend == 0).all())
        r.reset(torch.tensor([3]))
        self.assertFalse(r.stop_adjustment_active[3])
        self.assertTrue(r.stop_adjustment_active[2])

    def test_all_rewards_finite_and_ji22_terminal_separate(self):
        r = self.runtime
        r.ppo_iteration = 1800
        r.capture_recovery_active[:4] = True
        r.stop_adjustment_active[4:8] = True
        r.commands[8:12,1] = .8
        r.commands[12:,2] = .6
        r.command_targets[:] = r.commands[:,:3]
        r.base_lin_vel[:] = torch.randn(16,3)*.1
        r.dof_vel[:] = torch.randn(16,12)*.1
        r.rigid_body_states[:,:4,2] = .04
        r.contact_forces[:,:4,2] = 35
        self.assertEqual(len(r.reward_names),78)
        positive,negative = r.zeros(),r.zeros()
        # Clone because feet_air_time updates its event buffers during reward evaluation.
        expected = copy.deepcopy(r)
        for name,fn in zip(expected.reward_names,expected.reward_functions):
            value = fn()*expected.reward_scales[name]
            self.assertTrue(torch.isfinite(value).all(),name)
            if expected.reward_scales[name]>0:
                positive += value
            else:
                negative += value
        r.reset_buf[0] = True
        r.time_out_buf[1] = True
        r.reset_buf[1] = True
        r.compute_reward()
        reference = positive*torch.exp((negative/.02).clamp(-20,0))
        reference[0] += -.04
        torch.testing.assert_close(r.rew_buf,reference)

    def test_push_curriculum_and_capture_activation(self):
        r = self.runtime
        r.episode_length_buf[:] = 500
        calls = []
        r._velocity_writer = lambda ids: calls.append(ids.clone())
        r.ppo_iteration = 1000
        r._push_robots()
        self.assertTrue(calls)
        self.assertLessEqual(r.root_states[:,7:9].norm(dim=-1).max(),.800001)
        self.assertFalse(r.capture_recovery_active.any())
        r.ppo_iteration = 1800
        r._start_capture_recovery(torch.arange(16))
        self.assertTrue(r.capture_recovery_active.all())
        r.reset(torch.tensor([0,1]))
        self.assertFalse(r.capture_recovery_active[:2].any())
        self.assertTrue(r.capture_recovery_active[2:].all())


class CheckpointTest(unittest.TestCase):
    def test_source_name_conversion_and_critic_expansion(self):
        migrate = load_checkpoint_helper()
        old = {"history_encoder.0.weight":torch.randn(3,282),"target_encoder.0.weight":torch.randn(3,45),"prototypes.weight":torch.randn(32,16),"critic.0.weight":torch.randn(5,297)}
        current = {"estimator.encoder.0.weight":torch.empty(3,282),"estimator.target.0.weight":torch.empty(3,45),"estimator.proto.weight":torch.empty(32,16),"critic.0.weight":torch.empty(5,491)}
        result = migrate(old,current)
        torch.testing.assert_close(result["critic.0.weight"][:,:297],old["critic.0.weight"])
        self.assertEqual(result["critic.0.weight"][:,297:].count_nonzero(),0)
        torch.testing.assert_close(result["estimator.encoder.0.weight"],old["history_encoder.0.weight"])
        with self.assertRaisesRegex(RuntimeError,"incompatible"):
            migrate({"critic.0.weight":torch.zeros(5,50)},{"critic.0.weight":torch.zeros(5,491)})
        with self.assertRaisesRegex(RuntimeError,"missing"):
            migrate({},current)


if __name__ == "__main__":
    unittest.main()
