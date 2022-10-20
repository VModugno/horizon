import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils.actionManager import ActionManager
from horizon.problem import Problem
from horizon.rhc.PhaseManager import PhaseManager, Phase
from horizon.solvers import Solver
from horizon.rhc.model_description import FullModelInverseDynamics
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import time


def barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))


# set up problem
ns = 50
tf = 2.0  # 10s
dt = tf / ns

# set up solver
solver_type = 'ilqr'

transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

# set up model
path_to_examples = os.path.dirname('../../examples/')
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

base_init = np.array([0.0, 0.0, 0.505, 0.0, 0.0, 0.0, 1.0])
q_init = {'lf_haa_joint': 0.0,
          'lf_hfe_joint': 0.9,
          'lf_kfe_joint': -1.52,

          'lh_haa_joint': 0.0,
          'lh_hfe_joint': 0.9,
          'lh_kfe_joint': -1.52,

          'rf_haa_joint': 0.0,
          'rf_hfe_joint': 0.9,
          'rf_kfe_joint': -1.52,

          'rh_haa_joint': 0.0,
          'rh_hfe_joint': 0.9,
          'rh_kfe_joint': -1.52}

# set up model
prb = Problem(ns, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init)

model.setContactFrame('lh_foot', 'vertex', dict(vertex_frames=['lh_foot']))
model.setContactFrame('rh_foot', 'vertex', dict(vertex_frames=['rh_foot']))
model.setContactFrame('lf_foot', 'vertex', dict(vertex_frames=['lf_foot']))
model.setContactFrame('rf_foot', 'vertex', dict(vertex_frames=['rf_foot']))


model.q.setBounds(model.q0, model.q0, 0)
model.v.setBounds(np.zeros(model.nv), np.zeros(model.nv), 0)

model.q.setInitialGuess(model.q0)

for f_name, f_var in model.fmap.items():
    f_var.setInitialGuess([0, 0, kd.mass()/4])


if solver_type != 'ilqr':
    Transcriptor.make_method(transcription_method, prb, transcription_opts)

# contact velocity is zero, and normal force is positive
for i, frame in enumerate(contacts):
    FK = kd.fk(frame)
    DFK = kd.frameVelocity(frame, kd_frame)
    DDFK = kd.frameAcceleration(frame, kd_frame)

    p = FK(q=model.q)['ee_pos']
    v = DFK(q=model.q, qdot=model.v)['ee_vel_linear']
    a = DDFK(q=model.q, qdot=model.v)['ee_acc_linear']

    # kinematic contact
    contact = prb.createConstraint(f"{frame}_vel", v, nodes=[])

    # unilateral forces
    fcost = barrier(model.fmap[frame][2] - 10.0)  # fz > 10
    unil = prb.createIntermediateCost(f'{frame}_unil', fcost, nodes=[])

    # clearance
    z_des = prb.createParameter(f'{frame}_z_des', 1)
    clea = prb.createConstraint(f"{frame}_clea", p[2] - z_des, nodes=[])

    # go straight
    # p0 = FK(q=model.q0)['ee_pos']
    # cy = prb.createIntermediateResidual(f'{frame}_y', 2 * p0[1] - p[1], nodes=[])

    # contact_y.append(cy)
    # add to fn container
    # contact_constr.append(contact)
    # unilat_constr.append(unil)
    # clea_constr.append(clea)
    # zdes_params.append(z_des)

# cost
prb.createResidual("min_q", 0.05 * (model.q[7:] - model.q0[7:]))
prb.createIntermediateResidual("min_q_ddot", 1e-2 * model.a)
for f_name, f_var in model.fmap.items():
    prb.createIntermediateResidual(f"min_{f_var.getName()}", 1e-3 * f_var)

pm = PhaseManager(nodes=ns)

c_phases = dict()
for c in contacts:
    c_phases[c] = pm.addTimeline(f'{c}_timeline')


i = 0
for c in contacts:
    stance_phase = Phase(f'stance_{c}', 10)
    stance_phase.addConstraint(prb.getConstraints(f'{c}_vel'))
    c_phases[c].registerPhase(stance_phase)

    flight_phase = Phase(f'flight_{c}', 5)
    flight_phase.addVariableBounds(prb.getVariables(f'f_{c}'), [0, 0, 0])
    flight_phase.addConstraint(prb.getVariables(f'f_{c}'), [0, 0, 0])
    c_phases[c].registerPhase(flight_phase)

for name, timeline in c_phases.items():
    print('timeline:', timeline.name)

    for phase in timeline.registered_phases:
        print('    registered_phases', phase)

    for phase in timeline.phases:
        print('    phase', phase)
exit()
model.setDynamics()

for constr in prb.getConstraints():
    print(constr)


opts = {'ilqr.max_iter': 440,
        'ilqr.alpha_min': 0.1,
        'ilqr.huu_reg': 0.0,
        'ilqr.kkt_reg': 0.0,
        'ilqr.integrator': 'RK4',
        'ilqr.closed_loop_forward_pass': True,
        'ilqr.line_search_accept_ratio': 1e-4,
        'ilqr.kkt_decomp_type': 'ldlt',
        'ilqr.constr_decomp_type': 'qr',
        'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': ns * 1e-3,
        'ipopt.max_iter': 500,
        }

# todo if receding is true ....
solver_bs = Solver.make_solver('ilqr', prb, opts)

try:
    solver_bs.set_iteration_callback()
except:
    pass

scoped_opts_rti = opts.copy()
scoped_opts_rti['ilqr.enable_line_search'] = False
scoped_opts_rti['ilqr.max_iter'] = 1
solver_rti = Solver.make_solver('ilqr', prb, scoped_opts_rti)

t = time.time()
solver_bs.solve()
elapsed = time.time() - t
print(f'bootstrap solved in {elapsed} s')
try:
    solver_rti.print_timings()
except:
    pass
solution = solver_bs.getSolutionDict()

# =========================================================================

import subprocess, rospy
from horizon.ros import replay_trajectory

os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
rospy.loginfo("'spot' visualization started.")

repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), np.array([]), {k: None for k in contacts}, kd_frame,
                                           kd)
iteration = 0

rate = rospy.Rate(1 / dt)
flag_action = 1
forces = [prb.getVariables('f_' + c) for c in contacts]
nc = 4
while True:
    iteration = iteration + 1
    print(iteration)

    x_opt = solution['x_opt']
    u_opt = solution['u_opt']

    shift_num = -1
    xig = np.roll(x_opt, shift_num, axis=1)

    for i in range(abs(shift_num)):
        xig[:, -1 - i] = x_opt[:, -1]
    prb.getState().setInitialGuess(xig)

    uig = np.roll(u_opt, shift_num, axis=1)

    for i in range(abs(shift_num)):
        uig[:, -1 - i] = u_opt[:, -1]
    prb.getInput().setInitialGuess(uig)

    prb.setInitialState(x0=xig[:, 0])

    solver_rti.solve()
    solution = solver_rti.getSolutionDict()


    repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in model.fmap.items()}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    rate.sleep()
