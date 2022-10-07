from horizon.problem import Problem
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.rhc.model_description import *
import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.utils.actionManager import ActionManager, Step
import casadi as cs
import rospy, rospkg

# set up model
path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdf_path = rospkg.RosPack().get_path('hyqreal_description') + '/robots/hyqreal_with_InailArm.urdf'
urdf = open(urdf_path, 'r').read()
rospy.set_param('/robot_description', urdf)

base_init = np.array([0., 0., 0.53, 0., 0.0, 0.0, 1.])

q_init = {"lf_haa_joint": -0.2,
          "lf_hfe_joint": 0.75,
          "lf_kfe_joint": -1.5,
          "rf_haa_joint": -0.2,
          "rf_hfe_joint": 0.75,
          "rf_kfe_joint": -1.5,
          "lh_haa_joint": -0.2,
          "lh_hfe_joint": 0.75,
          "lh_kfe_joint": -1.5,
          "rh_haa_joint": -0.2,
          "rh_hfe_joint": 0.75,
          "rh_kfe_joint": -1.5,
                "z_joint_1": 0.0,
                "z_joint_2": 1.0,
                "z_joint_3": 1.0,
                "z_joint_4": 0.0,
                "z_joint_5": 0.0,
                "z_joint_6": 0.0}

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)

# definition of the problem --> required by the ti
N = 50
tf = 10.
dt = tf / N

prb = Problem(N, receding=True, casadi_type=cs.MX)
prb.setDt(dt)


# set up model
model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init)

ti = TaskInterface(prb=prb,
                   model=model)

ti.setTaskFromYaml(os.path.dirname(__file__) + '/rt2_walk.yaml')

f0 = np.array([0, 0, kd.mass()/4 * 9.8])
init_force = ti.getTask('joint_regularization')

init_force.setRef(1, f0)
init_force.setRef(2, f0)
init_force.setRef(3, f0)
init_force.setRef(4, f0)


# final_base_xy = ti.getTask('final_base_xy')
# final_base_xy.setRef([1, 0, 0, 0, 0, 0, 1])

final_base_xy = ti.getTask('final_arm_ee')
final_base_xy.setRef([1, 1, 0, 0, 0, 0, 1])

z_base = ti.getTask('z_arm_ee')
z_base.setRef([0, 0, 1, 0, 0, 0, 1])

opts = dict()
am = ActionManager(ti, opts)
am._trot([10, 40])

# todo: horrible API
# l_contact.setNodes(list(range(5)) + list(range(15, 50)))
# r_contact.setNodes(list(range(0, 25)) + list(range(35, 50)))


# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')

# === adding minimization of angular momentum ===
# cd_fun = ti.model.kd.computeCentroidalDynamics()
# h_lin, h_ang, dh_lin, dh_ang = cd_fun(q, v, a)
# ti.prb.createIntermediateResidual('min_angular_mom', 1e-1 * dh_ang)

q.setBounds(ti.model.q0, ti.model.q0, nodes=0)
v.setBounds(ti.model.v0, ti.model.v0, nodes=0)

q.setInitialGuess(ti.model.q0)

for cname, cforces in ti.model.cmap.items():
    for c in cforces:
        c.setInitialGuess(f0[:c.size1()])

replay_motion = True
plot_sol = False

# todo how to add?
ti.prb.createFinalConstraint('final_v', v)
# ================================================================
# ================================================================
# ===================== stuff to wrap ============================
# ================================================================
# ================================================================
import rospy
from horizon.ros import replay_trajectory

ti.finalize()
ti.bootstrap()
ti.load_initial_guess()
ti.resample(dt_res=0.001)
solution = ti.solution
ti.save_solution('/tmp/dioboy.mat')

if replay_motion:

    # single replay
    q_sol = solution['q_res']
    frame_force_mapping = {cname: solution[f.getName()+'_res'] for cname, f in model.fmap.items()}
    repl = replay_trajectory.replay_trajectory(0.001, model.kd.joint_names(), q_sol, frame_force_mapping, model.kd_frame, model.kd)
    repl.sleep(1.)
    repl.replay(is_floating_base=True)


# =========================================================================
if plot_sol:
    from horizon.utils import plotter
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import casadi as cs

    hplt = plotter.PlotterHorizon(prb, solution)
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


repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), np.array([]), {k: None for k in model.fmap.keys()},
                                           model.kd_frame, kd)
iteration = 0


flag_action = 1
forces = [f for _, f in ti.model.fmap.items()]

rate = rospy.Rate(1/dt)
while True:
    iteration = iteration + 1
    print(iteration)

    solution = ti.solution
    am.execute(solution)
    ti.rti()

    repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in ti.model.fmap.items()}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    rate.sleep()




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
