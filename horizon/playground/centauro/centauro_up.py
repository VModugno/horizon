import logging
import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.problem import Problem
from horizon.rhc.model_description import *
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.utils.actionManager import ActionManager, Step
from casadi_kin_dyn import py3casadi_kin_dyn
import casadi as cs
import rospy

rospy.init_node('centauro_up_node')
# set up model
path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdffile = os.path.join(path_to_examples, 'urdf', 'centauro.urdf')
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)

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

wheels = [f'j_wheel_{i + 1}' for i in range(4)]
q_init.update(zip(wheels, 4 * [0.]))

ankle_yaws = [f'ankle_yaw_{i + 1}' for i in range(4)]
# q_init.update(zip(ankle_yaws, 4 * [0.]))
q_init.update(dict(ankle_yaw_1=np.pi/4))
q_init.update(dict(ankle_yaw_2=-np.pi/4))
q_init.update(dict(ankle_yaw_3=-np.pi/4))
q_init.update(dict(ankle_yaw_4=np.pi/4))

q_init.update(fixed_joint_map)

base_init = np.array([0, 0, 0.718565, 0, 0, 0, 1])

# set up model description
urdf = urdf.replace('continuous', 'revolute')

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf, fixed_joints=fixed_joint_map)
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
joint_names = kd.joint_names()[2:]
q_init = {k: v for k, v in q_init.items() if k not in fixed_joint_map.keys()}

# set up problem
N = 50
tf = 10.0
dt = tf / N

prb = Problem(N, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

# set up model
model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init,
                                 fixed_joint_map=fixed_joint_map)
# else:
#     model = SingleRigidBodyDynamicsModel(problem=prb,
#                                          kd=kd,
#                                          q_init=q_init,
#                                          base_init=base_init,
#                                          contact_dict=contact_dict)


ti = TaskInterface(prb=prb,
                   model=model)

ti.setTaskFromYaml(os.path.dirname(__file__) + '/centauro_config.yaml')

f0 = np.array([0, 0, 110, 0, 0, 0])
init_force = ti.getTask('joint_regularization')
# init_force.setRef(1, f0)
# init_force.setRef(2, f0)

final_base_x = ti.getTask('final_base_xy')
final_base_x.setRef([0, 1.5, 0, 0, 0, 0, 1])


# final_base_y = ti.getTask('base_posture')
# final_base_y.setRef([0, 0, 0.718565, 0, 0, 0, 1])

opts = dict()
# am = ActionManager(ti, opts)
# am._walk([10, 40], [0, 3])
# am._step(Step(frame='contact_1', k_start=20, k_goal=30))
# am._step(Step(frame='contact_2', k_start=20, k_goal=30))

# todo: horrible API
# l_contact.setNodes(list(range(5)) + list(range(15, 50)))
# r_contact.setNodes(list(range(0, 25)) + list(range(35, 50)))


# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')

# adding minimization of angular momentum
# cd_fun = ti.model.kd.computeCentroidalDynamics()
# h_lin, h_ang, dh_lin, dh_ang = cd_fun(q, v, a)
# ti.prb.createIntermediateResidual('min_angular_mom', 0.1 * dh_ang)

q.setBounds(ti.model.q0, ti.model.q0, nodes=0)
v.setBounds(ti.model.v0, ti.model.v0, nodes=0)

q.setInitialGuess(ti.model.q0)

for cname, cforces in ti.model.cmap.items():
    for c in cforces:
        c.setInitialGuess(f0[:c.size1()])

replay_motion = True
plot_sol = not replay_motion

# todo how to add?
ti.prb.createFinalConstraint('final_v', v)
# ================================================================
# ================================================================
# ===================== stuff to wrap ============================
# ================================================================
# ================================================================
from horizon.ros import replay_trajectory

ti.finalize()
ti.bootstrap()
solution = ti.solution
ti.save_solution('/tmp/dioboy.mat')
ti.replay_trajectory()

# =========================================================================
