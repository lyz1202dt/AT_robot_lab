"""Low discrete rectangles from the Cyberstar flat robustness stage."""
import numpy as np
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def discrete_rectangles(difficulty, cfg):
    shape = tuple(int(size / cfg.horizontal_scale) for size in cfg.size)
    heights = np.zeros(shape,dtype=np.int16)
    unit = cfg.vertical_scale
    choices = np.array([-0.02,-0.01,0.01,0.02]) / unit
    for _ in range(20):
        width, length = np.random.choice(np.arange(10,20,4),2)
        x = np.random.choice(np.arange(0,max(1,shape[0]-width),4))
        y = np.random.choice(np.arange(0,max(1,shape[1]-length),4))
        heights[x:x+width,y:y+length] = int(np.random.choice(choices))
    platform = int(3 / cfg.horizontal_scale / 2)
    cx,cy = shape[0]//2,shape[1]//2
    heights[cx-platform:cx+platform,cy-platform:cy+platform] = 0
    return heights


@configclass
class CyberstarDiscreteTerrainCfg(HfTerrainBaseCfg):
    function = discrete_rectangles
