import time

from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions import integrators
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os, rospkg, argparse
from scipy.io import loadmat
from itertools import filterfalse

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

spot_actions = ('wheelie', 'jump_up', 'jump_forward', 'jump_on_wall', 'leap', 'jump_twist', 'walk')
spot_solvers = ('ipopt', 'ilqr', 'gnsqp')

parser = argparse.ArgumentParser(description='motion planning for the BostonDynamics quadruped robot!')
parser.add_argument('--action', '-a', help='choose which action spot will perform', choices=spot_actions, default=spot_actions[1])
parser.add_argument('--solver', '-s', help='choose which solver will be used', choices=spot_solvers, default=spot_solvers[0])
parser.add_argument('--replay', '-r', help='visualize the robot trajectory in rviz', action='store_true', default=False)
parser.add_argument("--codegen", '-c', type=str2bool, nargs='?', const=True, default=False, help="generate c++ code for faster solving")
parser.add_argument("--warmstart", '-w', type=str2bool, nargs='?', const=True, default=False, help="save solutions to mat file")

args = parser.parse_args()

action = args.action

if action == 'walk':

  import spot_walk
else:
  import spot_motions
