import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils.actionManager import ActionManager
import casadi as cs

# set up problem
ns = 50
tf = 10.0  # 10s
dt = tf / ns

# set up solver
solver_type = 'ipopt'
transcription_method = 'multiple_shooting'  # can choose between 'multiple_shooting' and 'direct_collocation'
transcription_opts = dict(integrator='RK4')  # integrator used by the multiple_shooting

# set up model
path_to_examples = os.path.abspath(__file__ + "/../../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdffile = os.path.join(path_to_examples, 'urdf', 'cogimon.urdf')
urdf = open(urdffile, 'r').read()

contacts = ['l_sole', 'r_sole']

base_init = np.array([0., 0., 0.96, 0., 0.0, 0.0, 1.])

q_init = {"LHipLat":       -0.0,
          "LHipSag":       -0.363826,
          "LHipYaw":       0.0,
          "LKneePitch":    0.731245,
          "LAnklePitch":   -0.307420,
          "LAnkleRoll":    0.0,
          "RHipLat":       0.0,
          "RHipSag":       -0.363826,
          "RHipYaw":       0.0,
          "RKneePitch":    0.731245,
          "RAnklePitch":   -0.307420,
          "RAnkleRoll":    -0.0,
          "WaistLat":      0.0,
          "WaistYaw":      0.0,
          "LShSag":        1.1717860 ,    #0.959931,   #   1.1717860
          "LShLat":        -0.059091562,    #0.007266,   #   -0.059091562
          "LShYaw":        -5.18150657e-02,    #-0.0,       #   -5.18150657e-02
          "LElbj":         -1.85118,    #-1.919862,  #  -1.85118
          "LForearmPlate": 0.0,
          "LWrj1":         -0.523599,
          "LWrj2":         -0.0,
          "RShSag":        1.17128697,  #0.959931,    #   1.17128697
          "RShLat":        6.01664139e-02,  #-0.007266,   #   6.01664139e-02
          "RShYaw":        0.052782481,  #-0.0,        #   0.052782481
          "RElbj":         -1.8513760,  #-1.919862,   #   -1.8513760
          "RForearmPlate": 0.0,
          "RWrj1":         -0.523599,
          "RWrj2":         -0.0}

problem_opts = {'ns': ns, 'tf': tf, 'dt': dt}

model_description = 'whole_body'
ti = TaskInterface(urdf, q_init, base_init, problem_opts, model_description, contacts=contacts, enable_torques=True, is_receding=True)

ti.loadPlugins(['horizon.rhc.plugins.contactTaskSpot'])
ti.setTaskFromYaml('config_walk.yaml')

f0 = np.array([0, 0, 315, 0, 0, 0])
init_force = ti.getTask('joint_regularization')
init_force.setRef(1, f0)
init_force.setRef(2, f0)

if solver_type != 'ilqr':
    th = Transcriptor.make_method(transcription_method, ti.prb, opts=transcription_opts)

final_base_x = ti.getTask('final_base_x')
final_base_x.setRef([1, 0, 0, 0, 0, 0, 1])

# final_base_y = ti.getTask('final_base_y')
# final_base_y.setRef([0, 1, 0, 0, 0, 0, 1])


opts = dict()
am = ActionManager(ti, opts)
am._walk([10, 40], [0, 1])

# todo: horrible API
# l_contact.setNodes(list(range(5)) + list(range(15, 50)))
# r_contact.setNodes(list(range(0, 25)) + list(range(35, 50)))

def compute_cop(frame, xmin, xmax, ymin, ymax):

    # world wrench
    world_wrench = ti.model.fmap[frame]

    fk = cs.Function.deserialize(ti.kd.fk(frame))
    R = cs.transpose(fk(q=ti.prb.getVariables('q'))['ee_rot'])

    local_wrench = cs.vertcat(R @ world_wrench[:3], R @ world_wrench[3:])

    M_cop = cs.DM(4, 6)
    M_cop[:, 2] = [xmin, -xmax, ymin, -ymax]
    M_cop[[0, 1], 4] = [1, -1]
    M_cop[[2, 3], 3] = [-1, 1]

    f_cop = M_cop @ local_wrench

    cop_cnsrt = ti.prb.createIntermediateConstraint(f'cop_{frame}', f_cop)
    cop_cnsrt.setLowerBounds(-np.inf * np.ones(4))

sole_dim = [-0.2, 0.2, -0.1, 0.1]
# sole_dim = [-0.05, 0.05, -0.02, 0.02]
compute_cop('l_sole', sole_dim[0], sole_dim[1], sole_dim[2], sole_dim[3])
compute_cop('r_sole', sole_dim[0], sole_dim[1], sole_dim[2], sole_dim[3])
# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')

import casadi as cs
cd_fun = cs.Function.deserialize(ti.kd.computeCentroidalDynamics())

# adding minimization of angular momentum
h_lin, h_ang, dh_lin, dh_ang = cd_fun(q, v, a)
ti.prb.createIntermediateResidual('min_angular_mom', 0.1 * dh_ang)

q.setBounds(ti.q0, ti.q0, nodes=0)
v.setBounds(ti.v0, ti.v0, nodes=0)

q.setInitialGuess(ti.q0)

forces = [ti.prb.getVariables('f_' + c) for c in contacts]

for f in forces:
    f.setInitialGuess(f0)


replay_motion = True
plot_sol = not replay_motion

# todo how to add?
ti.prb.createFinalConstraint('final_v', v)
# ================================================================
# ================================================================
# ===================== stuff to wrap ============================
# ================================================================
# ================================================================
import subprocess, rospy
from horizon.ros import replay_trajectory

solver_bs, solver_rti = ti.getSolver()
solver_bs.solve()
solution = solver_bs.getSolutionDict()

if replay_motion:
    os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
    subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=cogimon'])
    rospy.loginfo("'cogimon' visualization started.")

    # single replay
    q_sol = solution['q']
    frame_force_mapping = {contacts[i]: solution[forces[i].getName()] for i in range(len(contacts))}
    repl = replay_trajectory.replay_trajectory(dt, ti.kd.joint_names()[2:], q_sol, frame_force_mapping, ti.kd_frame, ti.kd)
    repl.sleep(1.)
    repl.replay(is_floating_base=True)
# =========================================================================
if plot_sol:
    from horizon.utils import plotter
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import casadi as cs

    hplt = plotter.PlotterHorizon(ti.prb, solution)
    hplt.plotVariables([elem.getName() for elem in forces], show_bounds=False, gather=2, legend=False, dim=[0, 1, 2])
    hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False, dim=[3, 4, 5])

    hplt.plotFunction('dynamics', show_bounds=True, legend=True, dim=range(6))
    hplt.plotVariable('v', show_bounds=True, legend=True, dim=range(6))

    # some custom plots to visualize the robot motion
    fig = plt.figure()
    fig.suptitle('Contacts')
    gs = gridspec.GridSpec(1, 2)
    i = 0
    for contact in contacts:
        ax = fig.add_subplot(gs[i])
        ax.set_title('{}'.format(contact))
        i += 1
        FK = cs.Function.deserialize(ti.kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']
        for dim in range(3):
            ax.plot(np.array([range(pos.shape[1])]).flatten(), np.array(pos[dim, :]).flatten())

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(ti.kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'plane_xy')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten())

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(ti.kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'plane_xz')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten())

    plt.show()


# repl = replay_trajectory.replay_trajectory(dt, ti.kd.joint_names()[2:], np.array([]), {k: None for k in contacts},
#                                            ti.kd_frame, ti.kd)
# iteration = 0
#
# solver_rti.solution_dict['x_opt'] = solver_bs.getSolutionState()
# solver_rti.solution_dict['u_opt'] = solver_bs.getSolutionInput()
#
# flag_action = 1
# forces = [ti.prb.getVariables('f_' + c) for c in contacts]
# nc = 4
# while True:
#     iteration = iteration + 1
#     print(iteration)
#     #
#     am.execute(solver_rti)
#     solver_rti.solve()
#     solution = solver_rti.getSolutionDict()
#
#     repl.frame_force_mapping = {contacts[i]: solution[forces[i].getName()][:, 0:1] for i in range(nc)}
#     repl.publish_joints(solution['q'][:, 0])
#     repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)




# todo this is important: what if I want to get the current position of a CartesianTask? do it!
# import casadi as cs
# FK = cs.Function.deserialize(ti.kd.fk('l_sole'))
# DFK = cs.Function.deserialize(ti.kd.frameVelocity('l_sole', ti.kd_frame))
#
# p_lf_start = FK(q=ti.q0)['ee_pos']
# v_lf_start = DFK(q=ti.q0, qdot=ti.v0)['ee_vel_linear']
#
# FK = cs.Function.deserialize(ti.kd.fk('l_sole'))
# DFK = cs.Function.deserialize(ti.kd.frameVelocity('l_sole', ti.kd_frame))
#
# rf_start = FK(q=ti.q0)['ee_pos']
# v = DFK(q=ti.q0, qdot=ti.v0)['ee_vel_linear']

# lf_ref = np.array([[0, 0, 0, 0, 0, 0, 1]]).T
# lf_ref[:3] = lf_start
# rf_ref = np.array([[0, 0, 0, 0, 0, 0, 1]]).T
# rf_ref[:3] = rf_start
# zero_vel_lf.setRef([0, 0, 0])
# zero_vel_rf.setRef([0, 0, 0])

# l_contact = ti.getTask('foot_contact_l_foot')