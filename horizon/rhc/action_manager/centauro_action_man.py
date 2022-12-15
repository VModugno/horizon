import logging
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
import phase_manager.pyphase as pyphase
import phase_manager.pymanager as pymanager


def barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))


def _trj(tau):
    return 64. * tau ** 3 * (1 - tau) ** 3

def compute_polynomial_trajectory(k_start, nodes, nodes_duration, p_start, p_goal, clearance, dim=None):

    if dim is None:
        dim = [0, 1, 2]

    # todo check dimension of parameter before assigning it

    traj_array = np.zeros(len(nodes))

    start = p_start[dim]
    goal = p_goal[dim]

    index = 0
    for k in nodes:
        tau = (k - k_start) / nodes_duration
        trj = _trj(tau) * clearance
        trj += (1 - tau) * start + tau * goal
        traj_array[index] = trj
        index = index + 1

    return np.array(traj_array)

# set up problem
ns = 80
tf = 8.0  # 10s
dt = tf / ns

# set up solver
solver_type = 'ilqr'

transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

# set up model
path_to_examples = os.path.dirname('../../examples/')
urdffile = os.path.join(path_to_examples, 'urdf', 'centauro.urdf')
urdf = open(urdffile, 'r').read()

contacts = ['contact_1', 'contact_2', 'contact_3', 'contact_4']

base_init = np.array([0, 0, 0.718565, 0, 0, 0, 1.0])

fixed_joint_map = {'torso_yaw': 0.00,   # 0.00,

                    'j_arm1_1': 1.50,   # 1.60,
                    'j_arm1_2': 0.1,    # 0.,
                    'j_arm1_3': 0.2,   # 1.5,
                    'j_arm1_4': -2.2,  # 0.3,
                    'j_arm1_5': 0.00,   # 0.00,
                    'j_arm1_6': -1.3,   # 0.,
                    'j_arm1_7': 0.0,    # 0.0,

                    'j_arm2_1': 1.50,   # 1.60,
                    'j_arm2_2': 0.1,    # 0.,
                    'j_arm2_3': -0.2,   # 1.5,
                    'j_arm2_4': -2.2,   #-0.3,
                    'j_arm2_5': 0.0,    # 0.0,
                    'j_arm2_6': -1.3,   # 0.,
                    'j_arm2_7': 0.0,    # 0.0,
                    'd435_head_joint': 0.0,
                    'velodyne_joint': 0.0,

                    'j_wheel_1': 0.0,
                    'j_wheel_2': 0.0,
                    'j_wheel_3': 0.0,
                    'j_wheel_4': 0.0,

                   'ankle_yaw_1': 0.0, #np.pi/4,
                   'ankle_yaw_2': 0.0, #-np.pi/4,
                   'ankle_yaw_3': 0.0, #-np.pi/4,
                   'ankle_yaw_4': 0.0, #np.pi/4,

                    # 'hip_yaw_1': -0.746,
                    # 'hip_pitch_1': -1.254,
                    # 'knee_pitch_1': -1.555,
                    # 'ankle_pitch_1': -0.3,
                    #
                    # 'hip_yaw_2': 0.746,
                    # 'hip_pitch_2': 1.254,
                    # 'knee_pitch_2': 1.555,
                    # 'ankle_pitch_2': 0.3,
                    #
                    # 'hip_yaw_3': 0.746,
                    # 'hip_pitch_3': 1.254,
                    # 'knee_pitch_3': 1.555,
                    # 'ankle_pitch_3': 0.3,
                    #
                    # 'hip_yaw_4': -0.746,
                    # 'hip_pitch_4': -1.254,
                    # 'knee_pitch_4': -1.555,
                    # 'ankle_pitch_4': -0.3,

                    }

# initial config
q_init = {
    'hip_yaw_1': -0.746,
    'hip_pitch_1': -1.254,
    'knee_pitch_1': -1.555,
    'ankle_pitch_1': -0.3,

    'hip_yaw_2': 0.746,
    'hip_pitch_2': 1.254,
    'knee_pitch_2': 1.555,
    'ankle_pitch_2': 0.3,

    'hip_yaw_3': 0.746,
    'hip_pitch_3': 1.254,
    'knee_pitch_3': 1.555,
    'ankle_pitch_3': 0.3,

    'hip_yaw_4': -0.746,
    'hip_pitch_4': -1.254,
    'knee_pitch_4': -1.555,
    'ankle_pitch_4': -0.3,
}

# wheels = [f'j_wheel_{i + 1}' for i in range(4)]
# q_init.update(zip(wheels, 4 * [0.]))

ankle_yaws = [f'ankle_yaw_{i + 1}' for i in range(4)]
q_init.update(zip(ankle_yaws, 4 * [0.]))
#
q_init.update(dict(ankle_yaw_1=np.pi/4))
q_init.update(dict(ankle_yaw_2=-np.pi/4))
q_init.update(dict(ankle_yaw_3=-np.pi/4))
q_init.update(dict(ankle_yaw_4=np.pi/4))
#
q_init.update(fixed_joint_map)

# set up model description
urdf = urdf.replace('continuous', 'revolute')

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf, fixed_joints=fixed_joint_map)
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
q_init = {k: v for k, v in q_init.items() if k not in fixed_joint_map.keys()}

# set up model
prb = Problem(ns, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init,
                                 fixed_joint_map=fixed_joint_map)

model.setContactFrame('contact_1', 'vertex', dict(vertex_frames=['contact_1']))
model.setContactFrame('contact_2', 'vertex', dict(vertex_frames=['contact_2']))
model.setContactFrame('contact_3', 'vertex', dict(vertex_frames=['contact_3']))
model.setContactFrame('contact_4', 'vertex', dict(vertex_frames=['contact_4']))

model.q.setBounds(model.q0, model.q0, 0)
model.v.setBounds(np.zeros(model.nv), np.zeros(model.nv), 0)

model.q.setInitialGuess(model.q0)

f0 = [0, 0, kd.mass()/4 * 9.8]
for f_name, f_var in model.fmap.items():
    f_var.setInitialGuess(f0)

if solver_type != 'ilqr':
    Transcriptor.make_method(transcription_method, prb, transcription_opts)

contact_pos = dict()
# contact velocity is zero, and normal force is positive
for i, frame in enumerate(contacts):
    FK = kd.fk(frame)
    DFK = kd.frameVelocity(frame, kd_frame)
    DDFK = kd.frameAcceleration(frame, kd_frame)

    p = FK(q=model.q)['ee_pos']
    p_rot = FK(q=model.q)['ee_rot']
    v = DFK(q=model.q, qdot=model.v)['ee_vel_linear']
    v_ang = DFK(q=model.q, qdot=model.v)['ee_vel_angular']
    a = DDFK(q=model.q, qdot=model.v)['ee_acc_linear']

    # rot_err = cs.sumsqr(p_rot[2, :2])
    # prb.createIntermediateCost(f'{frame}_rot', 1e-2*rot_err)

    # kinematic contact
    contact = prb.createConstraint(f"{frame}_vel", v, nodes=[])
    # unilateral forces
    fcost = barrier(model.fmap[frame][2] - 10.0)  # fz > 10
    unil = prb.createIntermediateCost(f'{frame}_unil', 1e1 * fcost, nodes=[])

    # clearance
    contact_pos[frame] = FK(q=model.q0)['ee_pos']
    z_des = prb.createParameter(f'{frame}_z_des', 1)

    clea = prb.createConstraint(f"{frame}_clea", p[2] - z_des, nodes=[])

    # lat_vel = cs.vertcat(v[0:2], v_ang)
    # vert = prb.createConstraint(f"{frame}_vert", lat_vel)

# pos_x_final_base = prb.createParameter("pos_x_final_base", 1)
# final_base_x
prb.createFinalConstraint(f'min_q0', model.q[0] - 2.)  #
# prb.createFinalConstraint(f'min_q0', model.q[0] - model.q0[0] - .5)  #
# prb.createFinalConstraint(f'min_q0', model.q[0] - pos_x_final_base) #  nodes=range(40, 50))
# prb.createFinalResidual(f'min_q0', 10000 * (model.q[0] - pos_x_final_base))
# pos_x_final_base.assign(model.q0[0] + .2)

# final_base_y
prb.createFinalResidual(f'min_q1', 1000 * (model.q[1] - model.q0[1]))
# final_base_rot
# prb.createFinalConstraint(f'min_rot', model.q[5])

# joint posture
# prb.createResidual("min_rot", 1e-3 * (model.q[3:4] - model.q0[3:4]))
# prb.createResidual("min_q", 1e-6 * (model.q[7:] - model.q0[7:]))
# prb.createFinalResidual("min_qf", 1e2 * (model.q[7:] - model.q0[7:]))
# prb.createResidual("min_q", 2. * (model.q[7:] - model.q0[7:]))
prb.createResidual("min_q", 1. * (model.q[7:] - model.q0[7:]))

# joint acceleration
prb.createIntermediateResidual("min_q_ddot", 0.02 * model.a)
# prb.createIntermediateResidual("min_q_ddot", 1e-5 * model.a)
# contact forces
for f_name, f_var in model.fmap.items():
    # prb.createIntermediateResidual(f"min_{f_var.getName()}", 0.01 * f_var)
    prb.createIntermediateResidual(f"min_{f_var.getName()}", 1e-8 * (f_var - f0))

cplusplus = True

opts =dict()
# opts['logging_level']=logging.DEBUG
if cplusplus:
    pm = pymanager.PhaseManager(ns)
else:
    pm = PhaseManager(nodes=ns, opts=opts)

c_phases = dict()
for c in contacts:
     c_phases[c] = pm.addTimeline(f'{c}_timeline')
#
i = 0
for c in contacts:
    # stance phase
    stance_duration = 6
    if cplusplus:
        stance_phase = pyphase.Phase(stance_duration, f"stance_{c}")
    else:
        stance_phase = Phase(f'stance_{c}', stance_duration)

    stance_phase.addConstraint(prb.getConstraints(f'{c}_vel'))
    stance_phase.addCost(prb.getCosts(f'{c}_unil'))
    c_phases[c].registerPhase(stance_phase)
#
#     # flight phase
    flight_duration = 6
    if cplusplus:
        flight_phase = pyphase.Phase(flight_duration, f"flight_{c}")
    else:
        flight_phase = Phase(f'flight_{c}', flight_duration)

    flight_phase.addVariableBounds(prb.getVariables(f'f_{c}'),  np.array([[0, 0, 0]] * flight_duration).T, np.array([[0, 0, 0]] * flight_duration).T)
    flight_phase.addConstraint(prb.getConstraints(f'{c}_clea'))

    # vertical take-off
    # flight_phase.addConstraint(prb.getConstraints(f'{c}_vert'))  # nodes=[0, 1, 2]

    z_trj = np.atleast_2d(compute_polynomial_trajectory(0, range(flight_duration), flight_duration, contact_pos[c], contact_pos[c], 0.03, dim=2))

    flight_phase.addParameterValues(prb.getParameters(f'{c}_z_des'), z_trj)
    c_phases[c].registerPhase(flight_phase)

for c in contacts:
    stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
    flight = c_phases[c].getRegisteredPhase(f'flight_{c}')
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)

# for c in contacts:
#     print(c_phases[c].getActivePhase())

################################################ CRAWLING ##############################################################
n_cycle = 4
initial_phase = 5

# todo
# lift_contacts = ['contact_1', 'contact_4', 'contact_2', 'contact_3']
lift_contacts = ['contact_4', 'contact_2', 'contact_3', 'contact_1']
for i in range(n_cycle):
    for pos, step in enumerate(lift_contacts):
        c_phases[step].addPhase(c_phases[step].getRegisteredPhase(f'flight_{step}'), initial_phase + pos)
    initial_phase += pos + 1

#

# lift_contact = 'contact_1'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 5)
# #
# lift_contact = 'contact_2'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 6)
# #
# lift_contact = 'contact_3'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 7)
# #
# lift_contact = 'contact_4'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 8)
#
# lift_contact = 'contact_1'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 9)
# #
# lift_contact = 'contact_2'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 10)
# #
# lift_contact = 'contact_3'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 11)
# #
# lift_contact = 'contact_4'
# c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), 12)

################################################ TROTTING ##############################################################

# range_trot_1 = range(4, 16, 2)
# range_trot_2 = range(5, 17, 2)
#
# lift_contact = 'contact_1'
# for i in range_trot_1:
#     c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)
# lift_contact = 'contact_2'
# for i in range_trot_2:
#     c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)
# lift_contact = 'contact_3'
# for i in range_trot_2:
#     c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)
# lift_contact = 'contact_4'
# for i in range_trot_1:
#     c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)


########################################################################################################################
# for name, timeline in c_phases.items():
    # print('timeline:', timeline.getName())
#
#     for phase in timeline.registered_phases:
#         print('    registered_phases', phase.getName())
# #
#     for phase in timeline.phases:
#         print('    phase', phase)

    # print(prb.getConstraints(f'{c}_vel').getName(), ":")
    # print(" nodes:", prb.getConstraints(f'{c}_vel').getNodes())
    # print(" bounds:", prb.getConstraints(f'{c}_vel').getLowerBounds())
    # print(prb.getCosts(f'{c}_unil').getName(), ":")
    # print(" nodes:", prb.getCosts(f'{c}_unil').getNodes())

# for c_name, c_item in prb.getConstraints().items():
#     print(c_item.getName())
#     print(c_item.getNodes())
# print("=========================")
# for c_name, c_item in prb.getCosts().items():
#     print(c_item.getName())
#     print(c_item.getNodes())
# for c_name, c_item in prb.getVariables().items():
#     print(c_item.getName())
#     print(c_item.getNodes())
#     print(c_item.getLowerBounds())
# exit()

model.setDynamics()

# ilqr.max_iter': 1033,
#   'ilqr.alpha_min': 0.001,
#   'ilqr.step_length': 1.0,
#   'ilqr.use_filter': True,
#   'ilqr.filter_gamma': .01,
#   'ilqr.filter_beta': .99,
#   'ilqr.hxx_reg': 1000.0,
#   'ilqr.hxx_reg_base': 0.0,
#   'ilqr.hxx_reg_growth_factor': 8.,
#   'ilqr.kkt_reg': 0.0,
#   'ilqr.integrator': 'RK4',
#   'ilqr.merit_der_threshold': 1e-6,
#   'ilqr.step_length_threshold': 1e-9,
#   'ilqr.line_search_accept_ratio': 1e-4,
#   'ilqr.kkt_decomp_type': 'ldlt',
#   'ilqr.constr_decomp_type': 'qr',

opts = {'ilqr.max_iter': 200,
        'ilqr.alpha_min': 0.01,
        'ilqr.step_length_threshold': 1e-9,
        'ilqr.line_search_accept_ratio': 1e-4,
        # 'ilqr.enable_gn': True,
        # 'ilqr.hxx_reg_base': 0.0,
        # 'ilqr.n_threads': 0
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

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), np.array([]), {k: None for k in contacts}, kd_frame,
                                           kd, fixed_joint_map=fixed_joint_map)
iteration = 0

rate = rospy.Rate(1 / dt)
flag_action = 1
forces = [prb.getVariables('f_' + c) for c in contacts]
nc = 4

elapsed_time_list = []
elapsed_time_solution_list = []

while iteration < 1000:
    iteration = iteration + 1
    print(iteration)

    # for cnsrt_name, cnsrt_item in prb.getConstraints().items():
    #     print(f"{cnsrt_name}:\n {cnsrt_item.getNodes()}")

    # for cost_name, cost_item in prb.getCosts().items():
    #     print(f"{cost_name}:\n {cost_item.getNodes()}")
    # print("================================================")

    # for var_name, var_item in prb.getVariables().items():
    #     print(f"{var_name}:\n {var_item.getUpperBounds()}")

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


    tic = time.time()
    pm._shift_phases()
    elapsed_time = time.time() - tic
    print('cycle:', elapsed_time)
    elapsed_time_list.append(elapsed_time)

    tic_solve = time.time()
    solver_rti.solve()
    elapsed_time_solving = time.time() - tic_solve
    print('solve:', elapsed_time_solving)
    elapsed_time_solution_list.append(elapsed_time_solving)
    solution = solver_rti.getSolutionDict()

    repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in model.fmap.items()}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    rate.sleep()


print("elapsed time resetting nodes:", sum(elapsed_time_list) / len(elapsed_time_list))
print("elapsed time solving:", sum(elapsed_time_solution_list) / len(elapsed_time_solution_list))



