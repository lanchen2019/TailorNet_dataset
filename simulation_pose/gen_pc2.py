import os
import glob
import os.path as osp
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')

from global_var import ROOT
from utils.rotation import get_Apose, interpolate_pose
from smpl_torch import SMPLNP_Lres
from utils.ios import save_pc2, read_pc2, read_obj, write_obj
from utils.part_body import part_body_faces


if __name__ == '__main__':
    datadir = '/home/dongxulin/205_2/chenlan/3D-R2N2/dataset/V1-0061_001993/'
    pklname = datadir + 'vibe_output.pkl'
    objlist = sorted(glob.glob(datadir + 'renderhuman/data_smooth_winsize4/cuthandsobj/*.obj'))
    dst_path = '0061_001993.pc2'

    verts = np.zeros((len(objlist), 6890, 3))
    for bi in tqdm(range(len(objlist))):
        vert, faces = read_obj(objlist[bi])
        verts[bi] = vert*1000
    
    write_obj(verts[0], faces, 'tpose.obj')
    verts = np.array(verts)
    save_pc2(verts, dst_path)
