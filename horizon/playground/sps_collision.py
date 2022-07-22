from horizon import problem
from horizon.utils import utils, kin_dyn, plotter, mat_storer, collision
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.solvers import solver
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import replay_trajectory
import os, time
import numpy as np
import casadi as cs
import rospkg, rospy

from srdfdom.srdf import SRDF

# orientation error
def rot_error_constr(R, Rdes):
    Re = R @ Rdes.T
    S = (Re - Re.T)/2
    r = cs.vec(S)
    err = r / cs.sqrt(1 + cs.trace(Re))
    return err

def rot_error_cost(R, Rdes):
    Re = R @ Rdes.T
    return cs.trace(np.eye(3) - Re)  # proportional to sin(th_err/2)^2

# options
solver_type = 'ilqr'
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')
tf = 4.0
n_nodes = 100
dt = tf / n_nodes

# load URDF
rospack = rospkg.RosPack()
pkgpath = rospack.get_path('mecademic_description')
# pkgpath = rospack.get_path('ModularBot_5DOF')
urdfpath = os.path.join(pkgpath, 'urdf', 'meca_500_r3_capsule.urdf.xacro')
srdfpath = os.path.join(pkgpath, 'srdf', 'meca_500_r3_capsule.srdf')

urdf = open(urdfpath, 'r').read()
rospy.set_param('robot_description', urdf)

# remove spheres from URDF
from urdf_parser_py import urdf as urdf_parser
robot = urdf_parser.Robot.from_xml_string(urdf)
for link in robot.links:
    if link.collisions:
        spheres = []
        for coll in link.collisions:
            if isinstance(coll.geometry, urdf_parser.Sphere):
                spheres.append(coll)
        for sphere in spheres:
            # link.collisions.remove(sphere)  # DOESN'T WORK!!!
            link.remove_aggregate(sphere)

urdf = robot.to_xml_string()

kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
nq = kindyn.nq()
nv = kindyn.nv()

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

# define dynamics
prb = problem.Problem(n_nodes) #, casadi_type=cs.SX)
q = prb.createStateVariable('q', nq)
q_dot = prb.createStateVariable('q_dot', nv)
q_ddot = prb.createInputVariable('q_ddot', nv)
x, x_dot = utils.double_integrator(q, q_dot, q_ddot)
prb.setDynamics(x_dot)
prb.setDt(dt)

# transcription
if solver_type != 'ilqr':
    Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# initial state and initial guess
q_init = np.zeros(nq)
q.setBounds(q_init, q_init, 0)
q_dot.setBounds(np.zeros(nv), np.zeros(nv), 0)
q.setInitialGuess(q_init)

# joint limits
q_max = kindyn.q_max()
q_min = kindyn.q_min()
q.setBounds(lb=q_min, ub=q_max, nodes=range(1, n_nodes+1))

# zero final velocity 
q_dot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=n_nodes)

# cartesian targets
frame_A = 'ee_link_A'
FK_A = kindyn.fk(frame_A)
DFK_A = kindyn.frameVelocity(frame_A, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

frame_B = 'ee_link_B'
FK_B = kindyn.fk(frame_B)
DFK_B = kindyn.frameVelocity(frame_B, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

# initial conditions arm A
p_A = FK_A(q=q)['ee_pos']
p_start_A = FK_A(q=q_init)['ee_pos']
o_A = FK_A(q=q)['ee_rot']
o_start_A = FK_A(q=q_init)['ee_rot']
print(o_start_A)

# initial conditions arm B
p_B = FK_B(q=q)['ee_pos']
p_start_B = FK_B(q=q_init)['ee_pos']
o_B = FK_B(q=q)['ee_rot']
o_start_B = FK_B(q=q_init)['ee_rot']
print(o_start_B)

# v_A = DFK_A(q=q, qdot=q_dot)['ee_vel_linear']
# v_B = DFK_B(q=q, qdot=q_dot)['ee_vel_linear']

p_tgt_A = p_start_A + np.array([0.6, -0.0, 0.0])
p_tgt_B = p_start_B + np.array([-0.1, +0.0, 0.0])

o_tgt_A = o_start_A
print(rot_error_constr(o_start_A, o_tgt_A))
o_tgt_B = o_start_B
print(rot_error_constr(o_start_B, o_tgt_B))

# final node constraints
final_node=n_nodes
prb.createFinalConstraint('ee_tgt_A', p_A - p_tgt_A)
prb.createFinalConstraint('ee_tgt_B', p_B - p_tgt_B)
q_dot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=final_node)
# prb.createFinalConstraint('ee_tgt_A_rot', rot_error_constr(o_A, o_tgt_A))
# prb.createFinalConstraint('ee_tgt_B_rot', rot_error_constr(o_B, o_tgt_B))

orrore = cs.Function('error', [q], [p_A - p_tgt_A, p_B - p_tgt_B, rot_error_constr(o_A, o_tgt_A), rot_error_constr(o_B, o_tgt_B)])

# collision
srdf = open(srdfpath, 'r').read()
use_fcl_wrapper = True
if use_fcl_wrapper:
    ch = cas_kin_dyn.CollisionHandler(kd=kindyn, srdf=srdf)
    collision_fn = ch.getDistanceFunction()
else:
    ch = collision.CollisionHandler(urdfstr=urdf, srdfstr=srdf, kindyn=kindyn)
    collision_fn = ch.get_function()

# collision_constr = prb.createConstraint('collision', collision_fn(q))
# collision_constr.setBounds(lb=np.zeros(collision_fn.size1_out(0)),
#                            ub=np.full(collision_fn.size1_out(0), np.inf))

collision_sx = collision_fn(q)
print(collision_sx.size1())

if solver_type == 'ilqr':
    coll_threshold = prb.createParameter('coll_threshold', 1)
    coll_threshold.assign(0.10)
    collision_barrier = cs.if_else(collision_sx > coll_threshold, 0, collision_sx - coll_threshold)
    coll_weight = prb.createParameter('coll_weight', dim=1)
    prb.createResidual('collision', coll_weight*collision_barrier)
    coll_weight.assign(1e2)

    # cost on orientation
    rot_weight = prb.createParameter('rot_weight', dim=1)
    prb.createResidual('rot_error_A', rot_weight*rot_error_cost(o_A, o_tgt_A))
    prb.createResidual('rot_error_B', rot_weight*rot_error_cost(o_B, o_tgt_B))
    rot_weight.assign(1)

else:
    collision_const = prb.createConstraint('collision_const', collision_sx)
    collision_const.setBounds(lb=np.zeros(collision_sx.size1()), ub=np.inf*np.ones(collision_sx.size1()))


##### Cost function #####
# penalize input
prb.createIntermediateResidual('min_u', np.sqrt(1e-1) * q_ddot)

# penalize joint velocity
# prb.createIntermediateResidual('min_qdot_linear', np.sqrt(1e4) * q_dot[0])
# prb.createIntermediateResidual('min_qdot_rotary', np.sqrt(1e4) * q_dot[7])
prb.createIntermediateResidual('min_qdot', np.sqrt(1) * q_dot)

# create solver
prb_solver = solver.Solver.make_solver(
    solver_type, 
    prb,
    opts={
        'ipopt.tol': 1e-4,
        'ipopt.max_iter': 2000,
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.linear_solver': 'ma57',
        'ilqr.codegen_enabled': True,
        'ilqr.codegen_workdir': '/home/edoardo/ilqr_codegen',
        'ilqr.alpha_min': 0.01,
        'ilqr.enable_gn': True,
        'ilqr.max_iter': 400,
        'ilqr.verbose': True,
        }
)

try:
    prb_solver.set_iteration_callback()
except:
    pass

# solver
tic = time.time()
prb_solver.solve()

if solver_type=='ilqr':
    prb.getState().setInitialGuess(prb_solver.x_opt)
    prb.getInput().setInitialGuess(prb_solver.u_opt)
    coll_threshold.assign(0.05)
    prb_solver.solve()

    prb.getState().setInitialGuess(prb_solver.x_opt)
    prb.getInput().setInitialGuess(prb_solver.u_opt)
    coll_threshold.assign(0.01)
    prb_solver.solve()

toc = time.time()
print('time elapsed solving:', toc - tic)

solution = prb_solver.getSolutionDict()

dt_sol = prb_solver.getDt()
total_time = sum(dt_sol)
print(f"total trajectory time: {total_time}")


collision_values = collision_fn(solution['q']).toarray().T

orror_values_A = orrore(solution['q'])[2].toarray().T
orror_values_B = orrore(solution['q'])[3].toarray().T

from matplotlib import pyplot as plt
plt.plot(collision_values)
plt.grid()
plt.show()

plt.figure()
plt.plot(orror_values_A)
plt.plot(orror_values_B)
plt.grid()
plt.show()

# visualize the robot in RVIZ
joint_list = joint_names
replay_trajectory(total_time / n_nodes, joint_list, solution['q']).replay(is_floating_base=False)