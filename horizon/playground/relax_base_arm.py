
from horizon.problem import Problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.ros import replay_trajectory
from horizon.transcriptions.transcriptor import Transcriptor
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.solvers import solver
import os, argparse
from itertools import filterfalse
import numpy as np
import casadi as cs


import rospkg, rospy

tf = 2.0
N = 50
dt = tf/N
solver_type = 'ilqr'
qinit = {
    'relax_arm1_joint1': 1.0,
    'relax_arm1_joint2': -1.8,
    'relax_arm1_joint4': 0.8,
}

# kin dyn
urdf_path = rospkg.RosPack().get_path('relax_urdf') + '/urdf/relax.urdf'
with open(urdf_path, 'r') as f:
    urdf = f.read()

# cartesio 
srdf_path = rospkg.RosPack().get_path('relax_srdf') + '/srdf/relax.srdf'
with open(srdf_path, 'r') as f:
    srdf = f.read()



rospy.set_param('robot_description', urdf)

kd = cas_kin_dyn.CasadiKinDyn(urdf)
kd_frame = cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

# joint names
joint_names = kd.joint_names()
if 'reference' in joint_names: joint_names.remove('reference')
if 'universe' in joint_names: joint_names.remove('universe')
print(joint_names)

prb = Problem(N=N)
q = prb.createStateVariable('q', kd.nq())
dq = prb.createStateVariable('dq', kd.nv())
u = prb.createInputVariable('ddq', kd.nv() - 4)  # no y, z, rotx, roty ( = 6 + 2 = 8 input size )
ddq = cs.vertcat(u[0], 0, 0, 0, 0, u[1], u[2:])

# dynamics
x, x_dot = utils.double_integrator_with_floating_base(q, dq, ddq)
prb.setDynamics(x_dot)
prb.setDt(dt)

# useful vars
vzero = np.zeros(kd.nv())

# initial state
qinit = kd.mapToQ(qinit)
q.setBounds(qinit, qinit, nodes=0)
dq.setBounds(vzero, vzero, nodes=0)

# guess
q.setInitialGuess(qinit)

# feasibility
# prb.createIntermediateConstraint('feas', ddq[1:5])

# transcription
if solver_type != 'ilqr':
    Transcriptor.make_method('multiple_shooting', prb)

# regularization
prb.createIntermediateResidual('regq', 1e-8*(q[6:] - qinit[6:]))
prb.createIntermediateResidual('regv', 2e-1*dq)
prb.createIntermediateResidual('rega', 4e-2*ddq)

# target
base_xy_tgt = np.array([0.0, 0.1])
# prb.createFinalConstraint('base_tgt', q[0:2] - base_xy_tgt)
# q[5].setInitialGuess(0.7)

# ee tracking
ee_fk = cs.Function.deserialize(kd.fk('relax_arm1_linkEndEffector'))
ee_pos = ee_fk(q=q)['ee_pos']
ee_rot = ee_fk(q=q)['ee_rot']

ee_tgt = np.array([0.0, 1.0, 1.1])
prb.createResidual('ee_pos', ee_pos - ee_tgt)

# final vel
prb.createFinalConstraint('final_v', dq)

# solver
solv = solver.Solver.make_solver(solver_type, prb, 
    {'ilqr.alpha_min': 0.01, 
    'ilqr.use_filter': False, 
    'ilqr.enable_line_search': True,
    'ilqr.step_length_threshold': 1e-9,
    'ilqr.line_search_accept_ratio': 0.0,
    'ilqr.max_iter': 200,
    'ilqr.verbose': True,
    'ilqr.enable_gn': True,
    'ilqr.rho_base': 1e-6,
    'ilqr.codegen_enabled': True, 
    'ilqr.codegen_workdir': '/tmp/relax_codegen'})

try:
    solv.set_iteration_callback()
except:
    pass

class CartesioSolver:
    
    def __init__(self, urdf, pb) -> None:
        print('created!')
        kd_frame = cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        self.base_fk = cs.Function.deserialize(kd.fk('base_link'))
        self.base_dfk = cs.Function.deserialize(kd.frameVelocity('base_link', kd_frame))
        self.task = pb.getTask('relax_arm1_linkEndEffector')

    def solve(self):
        print('solve!') 
        Tref = self.task.getPoseReference()[0]
        print(Tref)
        solv.solve()
        q = solv.getSolutionDict()['q'][:, 1]
        dq = solv.getSolutionDict()['dq'][:, 1]
        ddq = solv.getSolutionDict()['ddq'][:, 1]
        return  q[7:, 1], \
                dq[6:, 1], \
                self.base_fk(q=q)['ee_pos'].toarray(), \
                self.base_fk(q=q)['ee_rot'].toarray()



def create_cartesio_solver(urdf, ik_pb):
    return CartesioSolver(urdf, ik_pb)

# prb_dict = prb.save()
# prb_dict['dynamics'] = solv.dyn.serialize()

solv.solve()    

solution = solv.getSolutionDict()

from matplotlib import pyplot as plt

ee_err = ee_fk(q=solution['q'])['ee_pos'].toarray() - np.atleast_2d(ee_tgt).T

plt.figure()
plt.plot(solution['q'][[0, 1, 5], :].T)

plt.figure()
plt.plot(solution['dq'][[0, 1, 5], :].T)

plt.figure()
plt.plot(solution['ddq'][[0, 1, 5], :].T)

plt.figure()
plt.plot(solution['q'][7:, :].T)

plt.figure()
plt.plot(ee_err.T)

plt.show()
