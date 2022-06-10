from horizon.utils import plotter
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros import replay_trajectory
from horizon.solvers.solver import Solver
from horizon.rhc.taskInterface import TaskInterface
import numpy as np
import rospkg
import casadi as cs

urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()

ns = 50
tf = 8.0  # 10s
dt = tf / ns
problem_opts = {'ns': ns, 'tf': tf, 'dt': dt}

model_description = 'whole_body'

q_init = {}

for i in range(3):
    q_init[f'arm_{i + 1}_joint_2'] = -1.9
    q_init[f'arm_{i + 1}_joint_3'] = 2.30
    q_init[f'arm_{i + 1}_joint_5'] = -0.4

base_init = np.array([0, 0, 0.72, 0, 0, 0, 1])

# todo: this should not be in initialization
#  I should add contacts after the initialization, as forceTasks, no?

contacts = [f'arm_{i+1}_TCP' for i in range(3)]
ti = TaskInterface(urdf, q_init, base_init, problem_opts, model_description, contacts=contacts)

ptgt_final = [0., 0., 0.]
task_base_x = {'type': 'cartesian',
               'frame': 'base_link',
               'name': 'final_base_x',
               'dim': [0],
               'nodes': [ns],
               'weight': 1e3}

task_base_y = {'type': 'cartesian',
               'frame': 'base_link',
               'name': 'final_base_y',
               'dim': [1],
               'nodes': [ns],
               'fun_type': 'cost',
               'weight': 1e3}

# todo this should add the contacts tasks:
# for c in contacts:
#     task_contact = {'type': 'force',
#                     'frame': c,
#                     'name': 'force_' + c,
#                     'dim': [0, 1, 2]}
#     ti.setTask(task_contact)

ti.setTask(task_base_x)
ti.setTask(task_base_y)

task_base_x = ti.getTask('final_base_x')
task_base_x.setRef(ptgt_final)

task_base_y = ti.getTask('final_base_y')
task_base_y.setRef(ptgt_final)

# todo: next section to wrap up like the lines above
contacts = [f'arm_{i+1}_TCP' for i in range(3)]
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')
forces = [ti.prb.getVariables('f_' + c) for c in contacts]

q0 = ti.q0
v0 = ti.v0
f0 = np.array([0, 0, 300])

# final velocity
v.setBounds(v0, v0, nodes=ns)
# regularization costs

# base rotation
ti.prb.createResidual("min_rot", 1e-4 * (q[3:5] - q0[3:5]))

# joint posture
ti.prb.createResidual("min_q", 1e-1 * (q[7:] - q0[7:]))

# joint velocity
ti.prb.createResidual("min_v", 1e-2 * v)

# final posture
ti.prb.createFinalResidual("min_qf", 1e1 * (q[7:] - q0[7:]))

# regularize input
ti.prb.createIntermediateResidual("min_q_ddot", 1e0 * a)

# regularize forces
for f in forces:
    ti.prb.createIntermediateResidual(f"min_{f.getName()}", 1e-3 * (f - f0))


# costs and constraints implementing a gait schedule
com_fn = cs.Function.deserialize(ti.kd.centerOfMass())

# save default foot height
default_foot_z = dict()

# contact velocity is zero, and normal force is positive
for i, frame in enumerate(contacts):
    # fk functions and evaluated vars
    fk = cs.Function.deserialize(ti.kd.fk(frame))
    dfk = cs.Function.deserialize(ti.kd.frameVelocity(frame, ti.kd_frame))

    ee_p = fk(q=q)['ee_pos']
    ee_rot = fk(q=q)['ee_rot']
    ee_v = dfk(q=q, qdot=v)['ee_vel_linear']

    # save foot height
    default_foot_z[frame] = (fk(q=q0)['ee_pos'][2]).toarray()

    # vertical contact frame
    rot_err = cs.sumsqr(ee_rot[2, :2])
    ti.prb.createIntermediateCost(f'{frame}_rot', 1e4 * rot_err)

solver_type = 'ipopt'

if solver_type != 'ilqr':
    Transcriptor.make_method('multiple_shooting', ti.prb)

# set initial condition and initial guess
q.setBounds(q0, q0, nodes=0)
v.setBounds(v0, v0, nodes=0)

q.setInitialGuess(q0)

for f in forces:
    f.setInitialGuess(f0)

# ========================= set actions =====================================
# am = ActionManager(prb, urdf, kd, dict(zip(contacts, forces)), default_foot_z)

contact0 = {'type': 'contact',
               'frame': contacts[0],
               'name': 'contact_' + contacts[0]}

contact1 = {'type': 'contact',
               'frame': contacts[1],
               'name': 'contact_' + contacts[1]}

contact2 = {'type': 'contact',
               'frame': contacts[2],
               'name': 'contact_' + contacts[2]}

ti.setTask(contact0)
ti.setTask(contact1)
ti.setTask(contact2)

c_0 = ti.getTask('contact_arm_1_TCP')
c_1 = ti.getTask('contact_arm_2_TCP')
c_2 = ti.getTask('contact_arm_3_TCP')

step = range(10, 30)
contact_c_0 = [c_n for c_n in list(range(ns+1)) if c_n not in step]
print(contact_c_0)
c_0.setNodes(contact_c_0)
c_1.setNodes(range(ns + 1))
c_2.setNodes(range(ns + 1))



# print('CONSTRAINTS:')
# for cnsrt, obj in ti.prb.getConstraints().items():
#     print(cnsrt,':', obj.getNodes(), type(obj))
#
# print('COSTS:')
# for cnsrt, obj in ti.prb.getCosts().items():
#     print(cnsrt,':', obj.getNodes(), type(obj))
#
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


solver_bs = Solver.make_solver(solver_type, ti.prb, opts)
solver_rti = Solver.make_solver(solver_type, ti.prb, opts_rti)

solver_bs.solve()
solution = solver_bs.getSolutionDict()

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

## single replay
q_sol = solution['q']
frame_force_mapping = {contacts[i]: solution[forces[i].getName()] for i in range(3)}
repl = replay_trajectory.replay_trajectory(dt, ti.kd.joint_names()[2:], q_sol, frame_force_mapping, ti.kd_frame, ti.kd)
repl.sleep(1.)
repl.replay(is_floating_base=True)
exit()

plot_flag = True
if plot_flag:
    import matplotlib.pyplot as plt
    import matplotlib

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(ti.kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xy')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(ti.kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xz')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')

    hplt = plotter.PlotterHorizon(ti.prb, solution)
    hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
    hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
    matplotlib.pyplot.show()
