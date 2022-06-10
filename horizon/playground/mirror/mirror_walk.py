import copy

import matplotlib.pyplot
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs
from horizon.problem import Problem
from horizon.utils import utils, kin_dyn, plotter
from horizon.utils.actionManager import Step, ActionManager
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers.solver import Solver
from horizon.ros import replay_trajectory
import rospy, rospkg
import os


ns = 50
tf = 8.0  # 10s
dt = tf / ns

prb = Problem(ns, receding=True)
path_to_examples = os.path.dirname('../examples/')
solver_type = 'ipopt'

urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()
rospy.set_param('/robot_description', urdf)

contacts = [f'arm_{i+1}_TCP' for i in range(3)]

fixed_joint_map = None
kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

q_init = {}

for i in range(3):
    q_init[f'arm_{i+1}_joint_2'] = -1.9
    q_init[f'arm_{i+1}_joint_3'] = 2.30
    q_init[f'arm_{i+1}_joint_5'] = -0.4

base_init = np.array([0, 0, 0.72, 0, 0, 0, 1])

q_init = {k: v for k, v in q_init.items()}

q0 = kd.mapToQ(q_init)
q0[:7] = base_init

nq = kd.nq()
nv = kd.nv()
nc = len(contacts)
nf = 3

v0 = np.zeros(nv)
prb.setDt(dt)

# state and control vars
q = prb.createStateVariable('q', nq)
v = prb.createStateVariable('v', nv)
a = prb.createInputVariable('a', nv)
forces = [prb.createInputVariable('f_' + c, nf) for c in contacts]
fmap = {k: v for k, v in zip(contacts, forces)}

f0 = np.array([0, 0, 250])
# dynamics ODE
_, xdot = utils.double_integrator_with_floating_base(q, v, a)
prb.setDynamics(xdot)

# underactuation constraint
id_fn = kin_dyn.InverseDynamics(kd, contacts, kd_frame)
tau = id_fn.call(q, v, a, fmap)
prb.createIntermediateConstraint('dynamics', tau[:6])

# final goal (a.k.a. integral velocity control)
ptgt_final = [0., 0., 0.]
vmax = [0.05, 0.05, 0.05]
ptgt = prb.createParameter('ptgt', 3)

# goalx = prb.createFinalResidual("final_z",  1e3*(q[2] - ptgt[2]))
goalx = prb.createFinalConstraint("final_x", 1e3 * q[0] - ptgt[0])
goaly = prb.createFinalResidual("final_y", 1e3 * (q[1] - ptgt[1]))
# goalrz = prb.createFinalResidual("final_rz", 1e3 * (q[5] - ptgt[2]))
# base_goal_tasks = [goalx, goaly, goalrz]

# final velocity
v.setBounds(v0, v0, nodes=ns)
# regularization costs

# base rotation
prb.createResidual("min_rot", 1e-4 * (q[3:5] - q0[3:5]))

# joint posture
prb.createResidual("min_q", 1e-1 * (q[7:] - q0[7:]))

# joint velocity
prb.createResidual("min_v", 1e-2 * v)

# final posture
prb.createFinalResidual("min_qf", 1e1 * (q[7:] - q0[7:]))

# regularize input
prb.createIntermediateResidual("min_q_ddot", 1e0 * a)

# regularize forces
for f in forces:
    prb.createIntermediateResidual(f"min_{f.getName()}", 1e-3 * (f - f0))

# costs and constraints implementing a gait schedule
com_fn = cs.Function.deserialize(kd.centerOfMass())

# save default foot height
default_foot_z = dict()

# contact velocity is zero, and normal force is positive
for i, frame in enumerate(contacts):
    # fk functions and evaluated vars
    fk = cs.Function.deserialize(kd.fk(frame))
    dfk = cs.Function.deserialize(kd.frameVelocity(frame, kd_frame))

    ee_p = fk(q=q)['ee_pos']
    ee_rot = fk(q=q)['ee_rot']
    ee_v = dfk(q=q, qdot=v)['ee_vel_linear']

    # save foot height
    default_foot_z[frame] = (fk(q=q0)['ee_pos'][2]).toarray()

    # vertical contact frame
    rot_err = cs.sumsqr(ee_rot[2, :2])
    prb.createIntermediateCost(f'{frame}_rot', 1e4 * rot_err)

    # todo action constraints
    # kinematic contact
    # unilateral forces
    # friction
    # clearance
    # xy goal

am = ActionManager(prb, urdf, kd, dict(zip(contacts, forces)), default_foot_z)

if solver_type != 'ilqr':
    Transcriptor.make_method('multiple_shooting', prb)

# set initial condition and initial guess
q.setBounds(q0, q0, nodes=0)
v.setBounds(v0, v0, nodes=0)

q.setInitialGuess(q0)

for f in forces:
    f.setInitialGuess(f0)

# s1 = Step('arm_1_TCP', 5, 20, clearance=0.2)
# s2 = Step('arm_2_TCP', 35, 49, clearance=0.2)

# am.setStep(s1)
# am.setStep(s2)
am._walk([10, 200], step_pattern=[0, 2, 1], step_nodes_duration=10)
# am._trot([50, 100])
# am._jump([55, 65])

# create solver and solve initial seed
# print('===========executing ...========================')

opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 1e-3,
        'ipopt.max_iter': 1000,
        'error_on_fail': True,
        'ilqr.max_iter': 200,
        'ilqr.alpha_min': 0.01,
        'ilqr.use_filter': False,
        'ilqr.hxx_reg': 0.0,
        'ilqr.integrator': 'RK4',
        'ilqr.merit_der_threshold': 1e-6,
        'ilqr.step_length_threshold': 1e-9,
        'ilqr.line_search_accept_ratio': 1e-4,
        'ilqr.kkt_decomp_type': 'qr',
        'ilqr.constr_decomp_type': 'qr',
        'ilqr.verbose': True,
        'ipopt.linear_solver': 'ma57',
        }

opts_rti = opts.copy()
opts_rti['ilqr.enable_line_search'] = False
opts_rti['ilqr.max_iter'] = 4

solver_bs = Solver.make_solver(solver_type, prb, opts)
solver_rti = Solver.make_solver(solver_type, prb, opts_rti)

ptgt.assign(ptgt_final, nodes=ns)
solver_bs.solve()
solution = solver_bs.getSolutionDict()

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

## single replay
q_sol = solution['q']
frame_force_mapping = {contacts[i]: solution[forces[i].getName()] for i in range(nc)}
repl = replay_trajectory.replay_trajectory(dt, kd.joint_names()[2:], q_sol, frame_force_mapping, kd_frame, kd)
repl.sleep(1.)
repl.replay(is_floating_base=True)
exit()
# =========================================================================
repl = replay_trajectory.replay_trajectory(dt, kd.joint_names()[2:], np.array([]), {k: None for k in contacts},
                                           kd_frame, kd)
iteration = 0

if solver_type == 'ilqr':
    solver_rti.solution_dict['x_opt'] = solver_bs.getSolutionState()
    solver_rti.solution_dict['u_opt'] = solver_bs.getSolutionInput()

flag_action = 1
while True:

    # if flag_action == 1 and iteration > 50:
    #     flag_action = 0
    #     am._trot([40, 80])
    #
    # if iteration > 100:
    #     ptgt.assign([1., 0., 0], nodes=ns)

    # if iteration > 160:
    #     ptgt.assign([0., 0., 0], nodes=ns)

    # if iteration % 20 == 0:
    #     am.setStep(s_1)
    #     am.setContact(0, 'rh_foot', range(5, 15))
    #
    iteration = iteration + 1
    print(iteration)
    #
    am.execute(solver_rti)

    # if iteration == 10:
    #     am.setStep(s_lol)
    # for cnsrt_name, cnsrt in prb.getConstraints().items():
    #     print(cnsrt_name)
    #     print(cnsrt.getNodes())
    # if iteration == 20:
    #     am._jump(list(range(25, 36)))
    # solver_bs.solve()
    solver_rti.solve()
    solution = solver_rti.getSolutionDict()

    repl.frame_force_mapping = {contacts[i]: solution[forces[i].getName()][:, 0:1] for i in range(nc)}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
#

# set ROS stuff and launchfile
plot = True
#
if plot:
    import matplotlib.pyplot as plt

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xy')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xz')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')

    hplt = plotter.PlotterHorizon(prb, solution)
    hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
    hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
    matplotlib.pyplot.show()




