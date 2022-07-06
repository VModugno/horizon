import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils.actionManager import ActionManager

# set up problem
ns = 50
tf = 2.0  # 10s
dt = tf / ns

# set up solver
solver_type = 'ilqr'

# set up model
path_to_examples = os.path.dirname('../../../examples/')
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()

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

problem_opts = {'ns': ns, 'tf': tf, 'dt': dt}

model_description = 'whole_body'

# todo: wrong way of adding the contacts contacts=['lf_foot']
ti = TaskInterface(urdf, q_init, base_init, problem_opts, model_description)
ti.loadPlugins(['horizon.rhc.plugins.contactTaskSpot'])

ti.setTaskFromYaml('config_walk.yaml')

f0 = np.array([0, 0, 55])
contact1 = ti.getTask('joint_regularization')
contact1.setRef(1, f0)
contact1.setRef(2, f0)
contact1.setRef(3, f0)
contact1.setRef(4, f0)

opts = dict()
am = ActionManager(ti, opts)
am._walk([10, 200], [0, 2, 1, 3])



# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')

q.setBounds(ti.q0, ti.q0, nodes=0)
v.setBounds(ti.v0, ti.v0, nodes=0)

q.setInitialGuess(ti.q0)

forces = [ti.prb.getVariables('f_' + c) for c in contacts]
f0 = np.array([0, 0, 55])
for f in forces:
    f.setInitialGuess(f0)


# ================================================================
# ================================================================
# ===================== stuff to wrap ============================
# ================================================================
# ================================================================
import subprocess, rospy
from horizon.ros import replay_trajectory

# todo
# ptgt.assign(ptgt_final, nodes=ns)

solver_bs, solver_rti = ti.getSolver()
solver_bs.solve()
solution = solver_bs.getSolutionDict()

os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
rospy.loginfo("'spot' visualization started.")

# single replay
# q_sol = solution['q']
# frame_force_mapping = {contacts[i]: solution[forces[i].getName()] for i in range(nc)}
# repl = replay_trajectory.replay_trajectory(dt, kd.joint_names()[2:], q_sol, frame_force_mapping, kd_frame, kd)
# repl.sleep(1.)
# repl.replay(is_floating_base=True)
# exit()
# =========================================================================
repl = replay_trajectory.replay_trajectory(dt, ti.kd.joint_names()[2:], np.array([]), {k: None for k in contacts},
                         ti.kd_frame, ti.kd)
iteration = 0

solver_rti.solution_dict['x_opt'] = solver_bs.getSolutionState()
solver_rti.solution_dict['u_opt'] = solver_bs.getSolutionInput()

flag_action = 1
forces = [ti.prb.getVariables('f_' + c) for c in contacts]
nc = 4
while True:
    iteration = iteration + 1
    print(iteration)
    #
    am.execute(solver_rti)
    solver_rti.solve()
    solution = solver_rti.getSolutionDict()

    repl.frame_force_mapping = {contacts[i]: solution[forces[i].getName()][:, 0:1] for i in range(nc)}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)