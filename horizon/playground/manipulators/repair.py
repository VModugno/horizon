from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.tasks.cartesianTask import CartesianTask, Task
from horizon.ros import replay_trajectory
from horizon.solvers import Solver
import rospkg, rospy
import numpy as np
import casadi as cs
from horizon.transcriptions.transcriptor import Transcriptor

urdf_path = rospkg.RosPack().get_path('repair_urdf') + '/urdf/repair.urdf'
urdf = open(urdf_path, 'r').read()

solver_type = 'ipopt' # ilqr
transcription_method = 'multiple_shooting'  # can choose between 'multiple_shooting' and 'direct_collocation'
transcription_opts = dict(integrator='RK4')  # integrator used by the multiple_shooting

N = 50
tf = 2.0
dt = tf / N
nf = 3
problem_opts = {'ns': N, 'tf': tf, 'dt': dt}

model_description = 'whole_body'
q_init = {}


q_init[f'arm_1_joint_1'] = 0.
q_init[f'arm_1_joint_2'] = -0.76
q_init[f'arm_1_joint_3'] = 1.5
q_init[f'arm_1_joint_4'] = -1.3
q_init[f'arm_1_joint_5'] = 0.
q_init[f'arm_1_joint_6'] = -0.4
q_init[f'arm_1_joint_7'] = 0.
q_init[f'arm_2_joint_1'] = 0.
q_init[f'arm_2_joint_2'] = -0.76
q_init[f'arm_2_joint_3'] = -1.5
q_init[f'arm_2_joint_4'] = -1.3
q_init[f'arm_2_joint_5'] = 0.
q_init[f'arm_2_joint_6'] = -0.4
q_init[f'arm_2_joint_7'] = 0.

ti = TaskInterface(urdf, q_init, None, problem_opts, model_description)

cart = {'type': 'Cartesian',
        'frame': 'arm_1_tcp',
        'name': 'arm_1_tcp_ee',
        'indices': [0, 1, 2, 3, 4, 5, 6],
        'nodes': [N]}

ti.setTaskFromDict(cart)
ee_cart = ti.getTask('arm_1_tcp_ee')
ee_cart.setRef([0.5, -0.2, 0.5, 0, 0, 0, 1])
ee_cart.setRef([0.5, -0.2, 0.5, 0, 0.7071068, 0, 0.7071068 ])

q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')


# fk_tcp = cs.Function.deserialize(ti.kd.fk('arm_1_tcp'))
# ee_pos = fk_tcp(q=q)['ee_pos']
# ee_des = [0, 0, 1]
# ti.prb.createFinalConstraint('ee_tgt', ee_pos - ee_des)

q.setBounds(ti.q0, ti.q0, nodes=0)
v.setBounds(ti.v0, ti.v0, nodes=0)

q.setInitialGuess(ti.q0)

if solver_type != 'ilqr':
    th = Transcriptor.make_method(transcription_method, ti.prb, opts=transcription_opts)

# ===========================================================================================
# ================================== to wrap ================================================
# ===========================================================================================
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


solver_bs = Solver.make_solver(solver_type, ti.prb, opts)

solver_bs.solve()
solution = solver_bs.getSolutionDict()

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

## single replay


q_sol = solution['q']
repl = replay_trajectory.replay_trajectory(dt, ti.kd.joint_names()[1:], q_sol)
repl.sleep(1.)
repl.replay(is_floating_base=False)

# plot_flag = True
# if plot_flag:
#     import matplotlib.pyplot as plt
#     import matplotlib
#
#     plt.figure()
#     for frame in :
#         FK = cs.Function.deserialize(ti.kd.fk(contact))
#         pos = FK(q=solution['q'])['ee_pos']
#
#         plt.title(f'feet position - plane_xy')
#         plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
#         plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
#         plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')
#
#
#
#         plt.title(f'feet position - plane_xz')
#         plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
#         plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
#         plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')
#
#     hplt = plotter.PlotterHorizon(ti.prb, solution)
#     hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
#     hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
#     matplotlib.pyplot.show()
#
#
