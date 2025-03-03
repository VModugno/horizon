import time

from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions import integrators
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os, math
from itertools import filterfalse

# get path to the examples folder
path_to_examples = os.path.abspath(__file__ + "/../../../examples/")

# mat storer

file_name = os.path.splitext(os.path.basename(__file__))[0]
ms = mat_storer.matStorer(path_to_examples + f'{file_name}.mat')

# options
rviz_replay = True
plot_sol = True

solver_type = 'ipopt'
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')
load_initial_guess = False

tf = 2.0
n_nodes = 100
t_jump = (1.0, 1.5)

# load urdf
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

# parameters
n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3
dt = tf / n_nodes

# define dynamics
prb = problem.Problem(n_nodes)
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
q_ddot = prb.createInputVariable('q_ddot', n_v)
f_list = [prb.createInputVariable(f'f{i}', n_f) for i in range(n_c)]
x, x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
prb.setDynamics(x_dot)
prb.setDt(dt)

# contact map
contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
contact_map = dict(zip(contacts_name, f_list))

# initial state and initial guess
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   0.0, 0.9, -1.5202315,
                   0.0, 0.9, -1.5300265,
                   0.0, 0.9, -1.5253125])

q.setBounds(q_init, q_init, 0)
q_dot.setBounds(np.zeros(n_v), np.zeros(n_v), 0)

q.setInitialGuess(q_init)

for f in f_list:
    f.setInitialGuess([0, 0, 55])


th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# dynamic feasibility
id_fn = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
tau = id_fn.call(q, q_dot, q_ddot, contact_map)
prb.createIntermediateConstraint("dynamic_feasibility", tau[:6])


# final velocity is zero
prb.createFinalConstraint('final_velocity', q_dot)

# final pose
q_tgt = q_init.copy()
q_tgt[0] = 0
q_tgt[5] = math.sin(math.pi / 4)
prb.createFinalConstraint('q_fb', q[:6] - q_tgt[:6])

prb.createFinalCost('q_f', cs.sumsqr(10 * (q[7:] - q_tgt[7:])))

# contact handling
k_all = range(1, n_nodes + 1)
k_swing = list(range(*[int(t / dt) for t in t_jump]))
k_stance = list(filterfalse(lambda k: k in k_swing, k_all))
lifted_legs = contacts_name.copy()


# contact velocity is zero, and normal force is positive
for frame, f in contact_map.items():
    nodes = k_stance if frame in lifted_legs else k_all

    FK = cs.Function.deserialize(kindyn.fk(frame))
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    DDFK = cs.Function.deserialize(kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

    p = FK(q=q)['ee_pos']
    p_start = FK(q=q_init)['ee_pos']
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
    a = DDFK(q=q, qdot=q_dot)['ee_acc_linear']

    # node: on first swing node vel should be zero!
    prb.createConstraint(f"{frame}_vel", v, nodes=list(nodes) + [k_swing[0]])


# swing force is zero
for leg in lifted_legs:
    fzero = np.zeros(n_f)
    contact_map[leg].setBounds(fzero, fzero, nodes=k_swing)

# cost
# prb.createCost("min_rot", 10 * cs.sumsqr(q[3:6] - q_init[3:6]))
# prb.createCost("min_xy", 100 * cs.sumsqr(q[0:2] - q_init[0:2]))
prb.createCost("min_q", cs.sumsqr(1e-1 * (q[7:] - q_init[7:])))
# prb.createCost("min_q_dot", 1e-2 * cs.sumsqr(q_dot))
prb.createIntermediateCost("min_q_ddot", cs.sumsqr(1e-3 * (q_ddot)))
for f in f_list:
    prb.createIntermediateCost(f"min_{f.getName()}", cs.sumsqr(1e-3 * (f)))
# =============
# SOLVE PROBLEM
# =============
opts = dict()
opts['ipopt.tol'] = 0.001
opts['ipopt.constr_viol_tol'] = n_nodes * 1e-12
opts['ipopt.max_iter'] = 2000
opts['ipopt.linear_solver'] = 'ma57'

solver = solver.Solver.make_solver(solver_type, prb, opts)


t = time.time()
solver.solve()
elapsed = time.time() - t
print(f'solved in {elapsed} s')

solution = solver.getSolutionDict()
solution_constraints_dict = dict()

if isinstance(dt, cs.SX):
    ms.store({**solution, **solution_constraints_dict})
else:
    dt_dict = dict(dt=dt)
    ms.store({**solution, **solution_constraints_dict, **dt_dict})

# ========================================================
if plot_sol:

    hplt = plotter.PlotterHorizon(prb, solution)
    # hplt.plotVariables(show_bounds=True, legend=False)
    hplt.plotVariables([elem.getName() for elem in f_list], show_bounds=True, gather=2, legend=False)
    # hplt.plotFunction('inverse_dynamics', show_bounds=True, legend=True, dim=range(6))

    pos_contact_list = list()
    for contact in contacts_name:
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']
        plt.figure()
        plt.title(contact)
        for dim in range(n_f):
            plt.plot(np.array([range(pos.shape[1])]), np.array(pos[dim, :]), marker="x", markersize=3,
                     linestyle='dotted')

    plt.figure()
    for contact in contacts_name:
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'plane_xy')
        plt.scatter(np.array(pos[0, :]), np.array(pos[1, :]), linewidth=0.1)

    plt.figure()
    for contact in contacts_name:
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'plane_xz')
        plt.scatter(np.array(pos[0, :]), np.array(pos[2, :]), linewidth=0.1)

    plt.show()
# ======================================================

contact_map = dict(zip(contacts_name, [solution['f0'], solution['f1'], solution['f2'], solution['f3']]))

if rviz_replay:

    from horizon.ros.replay_trajectory import replay_trajectory
    import rospy

    try:
        # set ROS stuff and launchfile
        import subprocess

        # temporary add the example path to the environment
        os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
        subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
        rospy.loginfo("'spot' visualization started.")
    except:
        print('Failed to automatically run RVIZ. Launch it manually.')

    # remember to run a robot_state_publisher
    repl = replay_trajectory(dt, joint_names, solution['q'], contact_map,
                             cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

    repl.sleep(1.)
    repl.replay(is_floating_base=True)

