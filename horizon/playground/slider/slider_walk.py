import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.utils.actionManager import ActionManager, Step
from casadi_kin_dyn import py3casadi_kin_dyn
import casadi as cs
import rospy

# set up problem
ns = 50
tf = 10.0  # 10s
dt = tf / ns

# set up model
path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdffile = os.path.join(path_to_examples, 'urdf', 'slider_2_0_urdf.urdf')
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)


kd = py3casadi_kin_dyn.CasadiKinDyn(urdf)

base_init = np.array([0., 0., 0.5, 0., 0.0, 0.0, 1.])

q_init = {"Left_Roll":         0.,
          "Left_Pitch":        0.,
          "Left_Slide":        0.,
          "Left_Foot_Pitch":   0.,
          "Left_Foot_Roll":    0.,
          "Right_Roll":        0.,
          "Right_Pitch":       0.,
          "Right_Slide":       0.,
          "Right_Foot_Pitch":  0.,
          "Right_Foot_Roll":   0.,
         }

problem_opts = {'ns': ns, 'tf': tf, 'dt': dt}

model_description = 'whole_body'
ti = TaskInterface(urdf,
        q_init, base_init,
        problem_opts,
        model_description,
        is_receding=True)

ti.setTaskFromYaml(os.path.dirname(__file__) + '/slider_config.yaml')


f0 = np.array([0, 0, 110, 0, 0, 0])
init_force = ti.getTask('joint_regularization')
init_force.setRef(1, f0)
init_force.setRef(2, f0)


final_base_x = ti.getTask('final_base_x')
final_base_x.setRef([0, 0, 0, 0, 0, 0, 1])

# final_base_y = ti.getTask('final_base_y')
# final_base_y.setRef([0, 1, 0, 0, 0, 0, 1])



opts = dict()
am = ActionManager(ti, opts)
am._walk([10, 40], [0, 1])
# am._step(Step(frame='Left_Foot', k_start=20, k_goal=30))

# todo: horrible API
# l_contact.setNodes(list(range(5)) + list(range(15, 50)))
# r_contact.setNodes(list(range(0, 25)) + list(range(35, 50)))


# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')

import casadi as cs
cd_fun = ti.kd.computeCentroidalDynamics()

# adding minimization of angular momentum
# h_lin, h_ang, dh_lin, dh_ang = cd_fun(q, v, a)
# ti.prb.createIntermediateResidual('min_angular_mom', 0.1 * dh_ang)

q.setBounds(ti.q0, ti.q0, nodes=0)
v.setBounds(ti.v0, ti.v0, nodes=0)

q.setInitialGuess(ti.q0)

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
import subprocess, rospy
from horizon.ros import replay_trajectory

ti.finalize()
ti.bootstrap()
solution = ti.solution
ti.save_solution('/tmp/dioboy.mat')


# if replay_motion:
#     os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
#     subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=slider'])
#     rospy.loginfo("'cogimon' visualization started.")

if replay_motion:

    # single replay
    q_sol = solution['q']
    frame_force_mapping = {cname: solution[f.getName()] for cname, f in ti.model.fmap.items()}
    repl = replay_trajectory.replay_trajectory(dt, ti.kd.joint_names()[2:], q_sol, frame_force_mapping, ti.kd_frame, ti.kd)
    repl.sleep(1.)
    repl.replay(is_floating_base=True)
# =========================================================================
