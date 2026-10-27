"""03dog URDF spawning with its existing, separately reported FOOT bodies.

The CAD URDF already provides four fixed FOOT links. Preserve its masses,
inertias and collisions; Cyberstar's CSDog1 USD foot reconstruction is unnecessary.
"""

import isaaclab.sim as sim_utils


def spawn_dog03(prim_path, cfg, translation=None, orientation=None, **kwargs):
    return sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
