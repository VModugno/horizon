
from ctypes import util
from horizon.problem import Problem
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.solvers import Solver
import os
import numpy as np
from horizon.utils import utils, kin_dyn
import casadi as cs
import rospy
import logging


# set up model
path_to_examples = os.path.dirname(os.path.abspath(__file__)) + "/../examples"
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples


urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
print(path_to_examples, urdffile, os.path.dirname(__file__))
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)

base_init = np.array([0., 0., 0.96, 0., 0.0, 0.0, 1.])

q_init = {
}


contact_dict = {
    'l_sole': {
        'type': 'vertex',
        'vertex_frames': [
            'l_foot_lower_left_link',
            'l_foot_upper_left_link',
            'l_foot_lower_right_link',
            'l_foot_upper_right_link',
        ]
    },

    'r_sole': {
        'type': 'vertex',
        'vertex_frames': [
            'r_foot_lower_left_link',
            'r_foot_upper_left_link',
            'r_foot_lower_right_link',
            'r_foot_upper_right_link',
        ]
    }
}


# set up model
urdf = urdf.replace('continuous', 'revolute')

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)

# definition of the problem --> required by the ti
N = 50
tf = 3.0
dt = tf / N

prb = Problem(N, receding=True)

prb.setDt(dt)

q = prb.createStateVariable('q', kd.nq())
v_var = prb.createStateVariable('v', kd.nv())
a_var = prb.createStateVariable('a', kd.nv())
j_var = prb.createInputVariable('j', kd.nv())

v = v_var
a = a_var 
j = j_var

contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
forces = [prb.createStateVariable(f'force_{c}', 3) for c in contacts]
fdot_var = [prb.createInputVariable(f'fdot_{c}', 3) for c in contacts]
fdot = fdot_var

xdot = cs.vertcat(
    kd.qdot()(q, v),
    a,
    j,
    *fdot
)

prb.setDynamics(xdot)

# initial cond
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.9, -1.5238505,
                       0.0, 0.9, -1.5202315,
                       0.0, 0.9, -1.5300265,
                       0.0, 0.9, -1.5253125])
vzero = np.zeros(kd.nv())
fzero = np.zeros(3)

q.setBounds(q_init, q_init, 0)
v_var.setBounds(vzero, vzero, 0)
a_var.setBounds(vzero, vzero, 0)

q.setInitialGuess(q_init)
[f.setInitialGuess(np.array([0, 0, 50])) for f in forces]


# dynamic feasibility
id_fn = kin_dyn.InverseDynamics(kd, contacts, pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
tau = id_fn.call(q, v, a, {c: f for c, f in zip(contacts, forces)})
prb.createConstraint("dynamic_feasibility", tau[:6], nodes=range(1, N+1))

prb.createResidual('reg_q', q - q_init)
prb.createResidual('reg_v', v)
prb.createResidual('reg_a', a)
prb.createIntermediateResidual('reg_j', j)
[prb.createResidual(f'reg_{f.getName()}', 1e-3*f) for f in forces]
[prb.createIntermediateResidual(f'reg_{f.getName()}', f) for f in fdot]

prb.createFinalConstraint('final_v', v)

sol = Solver.make_solver('ilqr', prb, opts={
    'ilqr.max_iter': 100,
    'ilqr.verbose': True,
    # 'ilqr.log': True,
    'ilqr.svd_threshold': 1e-12,
    'ilqr.kkt_decomp_type': 'qr',
})

sol.solve()

from matplotlib import pyplot as plt
for k, v in sol.getSolutionDict().items():
    plt.figure()
    plt.plot(v.T)
    plt.title(k)

plt.show()