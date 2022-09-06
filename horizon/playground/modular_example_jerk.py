#!/usr/bin/env python3

from casadi.casadi import vertcat
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, mat_storer, resampler_trajectory
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils import plotter
from horizon.utils.plotter import PlotterHorizon
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os
import time
from horizon.ros import utils as horizon_ros_utils
import rospkg
from horizon.ros.replay_trajectory import *
import logging

from urdf_parser_py import urdf as urdf_parser

def compute_vel_max(m_R, body_part='chest', formula='use_energy'):
    # From ISO/TS 15066
    if body_part == "hands":
        m_H = 0.6 # kg. (effective mass of hands and fingers body part)
        k = 75000
        E_max = 0.49 # 2x because of transient contact!
        F_max = 2 * 140
    elif body_part == "face":
        m_H = 4.4 # kg. (effective mass of face body part)\
        k = 75000
        E_max = 0.11  # 2x because of transient contact!
        F_max = 2 * 65
    elif body_part == "chest":
        m_H = 40 # kg. (effective mass of face body part)\
        k = 25000
        E_max = 1.6  # 2x because of transient contact!
        F_max = 2 * 140
    elif body_part == "abdomen":
        m_H = 40 # kg. (effective mass of face body part)\
        k = 10000
        E_max = 2.4  # 2x because of transient contact!
        F_max = 2 * 110
    elif body_part == "back":
        m_H = 40 # kg. (effective mass of face body part)\
        k = 35000
        E_max = 2.5  # 2x because of transient contact!
        F_max = 2 * 210
    else:
        print("NOT HANDLED BODY PART")

    mu = m_H * m_R / (m_H + m_R)

    if formula == "use_force":
        v_max = F_max / cs.sqrt(mu * k)
    elif formula == "use_energy":
        v_max = cs.sqrt(2*E_max/mu)

    return v_max

rospack = rospkg.RosPack()
pkgpath = rospack.get_path('ModularBot_6DOF')
urdfpath = os.path.join(pkgpath, 'urdf', 'ModularBot.urdf')

urdf = open(urdfpath, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
nq = kindyn.nq()
nv = kindyn.nv()

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

trans = 'multiple_shooting'
# trans = 'direct_collocation'
solver_type = 'ipopt'
optimize_tf = False
optimize_dt = False
is_tau_input = False

k = 1  # 2
ns = k*50  # number of shooting intervals

# create problem
if optimize_tf:
    prb = problem.Problem(ns, logging_level=logging.DEBUG)
    tf = prb.createVariable("tf", 1)
    tf_min = 0.01
    tf_max = 20.0
    tf.setBounds(tf_min, tf_max)
    dt = tf/ns
elif optimize_dt:
    prb = problem.Problem(ns)
    dt = prb.createVariable("dt", 1, nodes=list(range(0, ns)))

    dt_min = [0.005] #[s]
    dt_max = [0.50] #[s]
    dt_init = [dt_min]
    dt.setBounds(dt_min, dt_max)
    dt.setInitialGuess(dt_init)
else:
    prb = problem.Problem(ns)
    tf = 3.0  # [s]
    dt = tf/ns

optimize_base_position = False
if optimize_base_position:
    x_offset = prb.createSingleVariable("x_offset", 1)
    x_min = 0.0
    x_max = 1.2
    x_offset.setBounds(x_min, x_max)
    x_offset.setInitialGuess(0.1)
    y_offset = prb.createSingleVariable("y_offset", 1)
    y_min = 0.0
    y_max = 0.6
    y_offset.setBounds(y_min, y_max)
    y_offset.setInitialGuess(0.1)

    # create state and input
    q = prb.createStateVariable("q", nq-2)
    qdot = prb.createStateVariable("qdot", nv-2)
    qddot = prb.createStateVariable("qddot", nv-2)
    q_aug = cs.vertcat(x_offset, y_offset, q)
    qdot_aug = cs.vertcat(cs.SX.zeros(2,1), qdot)
    qddot_aug = cs.vertcat(cs.SX.zeros(2,1), qddot)
else:
    q = prb.createStateVariable("q", nq)
    qdot = prb.createStateVariable("qdot", nv)
    qddot = prb.createStateVariable("qddot", nv)

FD = kindyn.aba()
ID = cs.Function.deserialize(kindyn.rnea())

# tau_input doesen't make sense. do not use
if is_tau_input:
    # tau as input
    tau = prb.createInputVariable("tau", nv)
    # forward dynamics
    qddot = FD(q=q, v=qdot, tau=tau)['a']
    # qddot = prb.createStateVariable("qddot", nv) ???
else:
    # jerk as input
    if optimize_base_position:
        qdddot = prb.createInputVariable("qdddot", nv-2)
        qdddot_aug = cs.vertcat(cs.SX.zeros(2,1), qdddot)
    else:
        qdddot = prb.createInputVariable("qdddot", nv)
        
x = prb.getState().getVars()
u = prb.getInput().getVars()

# specify derivative of state vector
xdot = cs.vertcat(qdot, qddot, qdddot)
prb.setDynamics(xdot)

# transcription
th = Transcriptor.make_method(trans, prb, dt, opts=dict(integrator='RK4'))


##### Constraints #####
# joint limits
q_min = []
q_max = []
qdot_lims = []
tau_lims = []
robot = urdf_parser.Robot.from_xml_string(urdf)
for joint_name in joint_names:
    for joint in robot.joints:
        if joint.name == joint_name:
            q_min.append(joint.limit.lower)
            q_max.append(joint.limit.upper)
            qdot_lims.append(joint.limit.velocity)
            tau_lims.append(joint.limit.effort)

qdot_lims = np.array(qdot_lims)
tau_lims = np.array(tau_lims)

if optimize_base_position:
    q.setBounds(lb=q_min[2:], ub=q_max[2:])
    qdot.setBounds(lb=(-qdot_lims[2:]).tolist(), ub=qdot_lims[2:].tolist())
else:
    q.setBounds(lb=q_min, ub=q_max)
    qdot.setBounds(lb=(-qdot_lims).tolist(), ub=qdot_lims.tolist())

if is_tau_input:
    tau.setBounds(lb=(-tau_lims).tolist(), ub=tau_lims.tolist())

# set initial state
# q0low = np.array([q_min[0], q_min[1], 0, 0, 0, 0, 0, 0])
# q0up = np.array([q_max[0], q_max[1], 0, 0, 0, 0, 0, 0])
# print(q0low)
# print(q0up)
q0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# q0 = np.array([0, 0, 0, 0, 0, 0])
q.setBounds(lb=q0, ub=q0, nodes=0)
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=0)
    qddot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=0)
else:
    q.setBounds(lb=q0, ub=q0, nodes=0)
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)
    qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)

# zero final velocity 
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=ns)
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=ns)
# prb.createFinalConstraint('qdot_final', qdot)

# zero final acceleration
if optimize_base_position:
    qddot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=ns)
else:
    qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=ns)
# prb.createFinalConstraint('qddot_final', qddot)

if is_tau_input:
    tau_start = ID(q=q0, v=np.zeros(nv), a=np.zeros(nv))['tau'].toarray()
    tau.setInitialGuess(tau_start, nodes=0) 
    tau.setBounds(lb=tau_start, ub=tau_start, nodes=0)
    tau_final = ID(q=q, v=np.zeros(nv), a=np.zeros(nv))['tau']
    tau.setBounds(lb=tau_final, ub=tau_final, nodes=ns-1)
    # prb.createConstraint('tau_final', tau - tau_final, nodes=ns-1)

else:
    # # zero initial jerk
    # qdddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)
    # # zero final jerk
    # qdddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=ns-1)

    # add inverse dynamics as constraint
    if optimize_base_position:    
        tau = ID(q=q_aug, v=qdot_aug, a=qddot_aug)['tau']
    else:
        tau = ID(q=q, v=qdot, a=qddot)['tau']
                
    tau_cnstrnt = prb.createIntermediateConstraint("inverse_dynamics", tau, bounds=dict(lb=-tau_lims, ub=tau_lims))


# cartesian target
frame = 'pen_A'
FK = cs.Function.deserialize(kindyn.fk(frame))
DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

# base offset
# x_input = cs.SX.sym('x_input')
# y_input = cs.SX.sym('y_input')
# z_input = cs.SX.sym('z_input')
# vector = cs.SX.sym('vector', 3, 1)
# make_vector = cs.Function('make_vector',[x_input,y_input, z_input], [vector], ['x_off','y_off','z_off'], ['vector'])
# base_offset = make_vector(x_off=x_offset, y_off=y_offset, z_off=0.0)['vector']
# base_offset = cs.vertcat(x_offset, y_offset, 0.0)
base_offset = cs.vertcat(0.0, 0.0, 0.0)

if optimize_base_position:
    p = FK(q=q_aug)['ee_pos']
else:
    p = FK(q=q)['ee_pos']
qinit = np.array([0, 0, 0, 0, 0, 0, 0, 0])
p_start = FK(q=qinit)['ee_pos'].toarray()

p_tgt = p_start.copy()
p_tgt[0] = 0.5
p_tgt[1] = 0.2
p_tgt[2] = 0.0

if optimize_base_position:
    v = DFK(q=q_aug, qdot=qdot_aug)['ee_vel_linear']
else:
    v = DFK(q=q, qdot=qdot)['ee_vel_linear']

homing_node = 20
prb.createConstraint('ee_tgt_1', p + base_offset - p_tgt, nodes=k*homing_node)

if optimize_base_position:
    quat = FK(q=q_aug)['ee_rot']
else:
    quat = FK(q=q)['ee_rot']    

quat_start = FK(q=qinit)['ee_rot'].toarray()
quat_tgt = quat_start.copy()

z_versor = quat[:,2]
plane_normal = np.array([0,0,1])
dot_product = cs.dot(plane_normal, z_versor)

# the z of the end effector frame should point towards the plane. With the bounds we can relax this a bit
prb.createConstraint('orientation_constraint_1', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*homing_node)
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=k*homing_node)
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*homing_node)

# we do the same for nodes 2
p_tgt[2] += 0.3
prb.createConstraint('ee_tgt_2', p + base_offset - p_tgt, nodes=k*(5+homing_node))
prb.createConstraint('orientation_constraint_2', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(5+homing_node))
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=k*(5+homing_node))
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(5+homing_node))

# we do the same for nodes 3
p_tgt[1] -= 0.4
prb.createConstraint('ee_tgt_3', p + base_offset - p_tgt, nodes=k*(10+homing_node))
prb.createConstraint('orientation_constraint_3', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(10+homing_node))
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=k*(10+homing_node))
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(10+homing_node))

# we do the same for nodes 4
p_tgt[2] -= 0.3
prb.createConstraint('ee_tgt_4', p + base_offset - p_tgt, nodes=k*(15+homing_node))
prb.createConstraint('orientation_constraint_4', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(15+homing_node))
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=k*(15+homing_node))
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(15+homing_node))

# we do the same for nodes 5
p_tgt[2] += 0.3
prb.createConstraint('ee_tgt_5', p + base_offset - p_tgt, nodes=k*(20+homing_node))
prb.createConstraint('orientation_constraint_5', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(20+homing_node))
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=k*(20+homing_node))
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(20+homing_node))

# we do the same for nodes 6
p_tgt[1] += 0.4
prb.createConstraint('ee_tgt_6', p + base_offset - p_tgt, nodes=k*(25+homing_node))
prb.createConstraint('orientation_constraint_6', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(25+homing_node))
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=k*(25+homing_node))
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(25+homing_node))

# we do the same for nodes 7
p_tgt[2] -= 0.3
prb.createConstraint('ee_tgt_7', p + base_offset - p_tgt, nodes=k*(30+homing_node))
prb.createConstraint('orientation_constraint_7', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(30+homing_node))
if optimize_base_position:
    qdot.setBounds(lb=np.zeros(nv-2), ub=np.zeros(nv-2), nodes=k*(30+homing_node))
else:
    qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(30+homing_node))

if False: # for quaternions
    # This impose a constraint on the orientation of the end-effector. the "pen" shoul point towards the plane
    q_x = q[0]
    q_y = q[1]
    q_z = q[2]
    q_w = q[3]
    q_x_tgt = q_tgt[0]
    q_y_tgt = q_tgt[1]
    q_z_tgt = q_tgt[2]
    q_w_tgt = q_tgt[3]
    q_z_tgt = 0.0
    q_w_tgt = 0.0
    prb.createFinalConstraint('ee_tgt_rot_z', q_z - q_z_tgt)
    prb.createFinalConstraint('ee_tgt_rot_w', q_w - q_w_tgt)

# max joint vel
# qdot.setBounds(lb=np.full(nv, -0.2), ub=np.full(nv, 0.2), nodes=range(1, ns))

# obstacle
# sphere_pos = np.array([0.4, 0.0, 1.5])
# sphere_r = 0.30
# obs = prb.createIntermediateConstraint('obstacle', 
#                 cs.sumsqr(sphere_pos - p),
#                 bounds=dict(lb=sphere_r**2))


# Compute moving mass in an ugly way :(
moving_mass = 0.0
moving_links = { k: v for k, v in robot.link_map.items()\
     if k not in ['world', 'base_link', 'J1_A_stator', 'pen_A', 'virtual_link'] }
for link_name, link_object in moving_links.items():
    if link_object.inertial is not None:
        moving_mass += link_object.inertial.mass
# From ISO/TS 15066
m_R = moving_mass / 2

# Compute max safe velocity
v_max = compute_vel_max(m_R, body_part='chest', formula='use_energy')

# Limit on max velocity
prb.createConstraint("v_max", cs.sumsqr(v), bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, v_max**2)))
#prb.createConstraint("v_max", cs.sumsqr(v) - v_max**2, bounds=dict(lb=np.full(1, -cs.inf), ub=np.full(1, 0.0)))

##### Cost function #####
# Cost
if optimize_tf:
    prb.createIntermediateCost('min_f', 100 * tf)
    prb.createIntermediateCost('min_u', 1 * cs.sumsqr(qdddot))
elif optimize_dt:
    prb.createIntermediateCost('min_f', 1000 * cs.sum2(dt))
    prb.createIntermediateCost('min_u', 1 * cs.sumsqr(qdddot))
else:
    # penalize input
    prb.createIntermediateCost('min_u', 1*cs.sumsqr(qdddot))
    #prb.createCostFunction('min_qdd', 1*cs.sumsqr(qddot))


# create solver
prb_solver = solver.Solver.make_solver(
    solver_type, 
    prb, 
    dt, 
    opts={
        'ipopt.tol': 1e-4,
        'ipopt.max_iter': 2000
        }
    )

# solver
tic = time.time()
prb_solver.solve()
toc = time.time()
print('time elapsed solving:', toc - tic)

solution = prb_solver.getSolutionDict()

if optimize_tf:
    tf = solution["tf"].item()
    print(f"Tf: {tf}")
elif optimize_dt:
    tf = np.sum(solution["dt"])
    print(f"Tf: {tf}")

if optimize_base_position:
    x_offset = solution["x_offset"].item()
    print(f"x_offset: {x_offset}")
    y_offset = solution["y_offset"]
    print(f"y_offset: {y_offset}")
    
plot_all = True
if plot_all:
    # plotter.PlotterHorizon(prb, solution).plotFunction('obstacle')
    # plt.figure()
    # psol = FK(q=solution['q'])['ee_pos'].toarray()
    # vsol = DFK(q=solution['q'], qdot=solution['qdot'])['ee_vel_linear'].toarray()
    # plt.plot(psol[1], psol[2], 'o-')
    # circle1 = plt.Circle(sphere_pos[[1, 2]], sphere_r, color='r')
    # plt.gca().add_patch(circle1)
    # plt.figure()
    # plt.plot(vsol[0], vsol[1], vsol[2], 'o-')

    # plotter.PlotterHorizon(prb, solution).plotVariable('qdot')
    # if is_tau_input:
    #     qddotsol = FD(q=solution['q'][:,:-1], v=solution['qdot'][:,:-1], tau=solution['tau'])['a'].toarray()
    #     plotter.PlotterHorizon(prb, solution).plotVariable('tau')
    #     plt.figure()
    #     # plt.plot(qddotsol.transpose(), range(0, ns))
    #     plt.plot(range(0, ns), qddotsol[0])
    #     plt.plot(range(0, ns), qddotsol[1])
    #     plt.plot(range(0, ns), qddotsol[2])
    #     plt.plot(range(0, ns), qddotsol[3])
    #     plt.plot(range(0, ns), qddotsol[4])
    #     plt.plot(range(0, ns), qddotsol[5])
    # else:
    #     plotter.PlotterHorizon(prb, solution).plotVariable('qddot')
    #     plotter.PlotterHorizon(prb, solution).plotVariable('qdddot')

    hplt = PlotterHorizon(prb, solution)
    # hplt.plotVariables(legend=False)
    hplt.plotVariable('q', show_bounds=True)
    hplt.plotVariable('qdot', show_bounds=True)
    hplt.plotVariable('qddot', show_bounds=True)
    hplt.plotVariable('qdddot', show_bounds=True)
    
    if optimize_tf:
        hplt.plotVariable('tf', markers='x', show_bounds=False)
    if optimize_base_position:
        hplt.plotVariable('x_offset', markers='x', show_bounds=False)
        hplt.plotVariable('y_offset', markers='x', show_bounds=False)
    # hplt.plotFunctions(legend=False)
    hplt.plotFunction('v_max', show_bounds=True)
    # plt.axhline(y=v_max**2, color='k', linestyle='--')
    hplt.plotFunction('inverse_dynamics', show_bounds=False)
    
    plt.show()


q_hist = solution["q"]
qdot_hist = solution["qdot"]
qddot_hist = solution["qddot"]
qdddot_hist = solution["qdddot"]

if optimize_base_position:
    x_hist = np.full((1,ns+1), x_offset)
    y_hist = np.full((1,ns+1), y_offset)
    q_aug_hist = np.vstack([x_hist, y_hist, q_hist])

resample = False
if optimize_dt or resample:
    #resample trajectory
    L = 0.5*cs.dot(qdot, qdot)  # Objective term
    dae = {'x': x, 'p': qdddot, 'ode': xdot, 'quad': L}

    dt = 0.01
    if optimize_dt:
        dt_hist = solution["dt"]
        dt_old = dt_hist.flatten()
    else:
        dt_old = tf/ns
    
    state_res = resampler_trajectory.resampler(np.array(cs.vertcat(q_hist, qdot_hist, qddot_hist)), qdddot_hist, dt_old, dt, dae)
    q_res = state_res[0:nq, :]
    qdot_res = state_res[nq:2*nq, :]
    qddot_res = state_res[2*nq:3*nq, :]

    qdddot_res = resampler_trajectory.resample_input(qdddot_hist, dt_old, dt)

    PLOTS = True
    if PLOTS:
        time = np.arange(0.0, state_res.shape[1]*dt, dt)

        plt.figure()
        for i in range(nq):
            plt.plot(time, q_res[i, :])
        plt.suptitle('$\mathrm{q}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[m]}$', size=20)

        plt.figure()
        for i in range(nq):
            plt.plot(time, qdot_res[i, :])
        plt.suptitle('$\mathrm{qdot}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[m]}$', size=20)

        plt.figure()
        for i in range(nq):
            plt.plot(time, qddot_res[i, :])
        plt.suptitle('$\mathrm{qddot}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[m]}$', size=20)

        plt.figure()
        for i in range(nq):
            plt.plot(time[:-1], qdddot_res[i, :])
        plt.suptitle('$\mathrm{qdddot}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[m]}$', size=20)

        tau_res = ID(q=q_res, v=qdot_res, a=qddot_res)['tau'].toarray()
        plt.figure()
        for i in range(nq):
            plt.plot(time, tau_res[i, :])
        plt.suptitle('$\mathrm{tau}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[m]}$', size=20)

        v_res = DFK(q=q_res, qdot=qdot_res)['ee_vel_linear'].toarray()
        v_norm = np.sqrt(np.sum(np.square(v_res), axis=0))
        plt.figure()
        plt.plot(time, v_norm)
        plt.axhline(y=v_max, color='k', linestyle='--')
        plt.suptitle('$\mathrm{linear velocity}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[m]}$', size=20)

        plt.show()
    
    replay_trajectory(dt, kindyn.joint_names()[1:], q_res).replay(is_floating_base=False)
else:
    if optimize_base_position:
        replay_trajectory(tf/ns, kindyn.joint_names()[1:], q_aug_hist).replay(is_floating_base=False)
    else:
        replay_trajectory(tf/ns, kindyn.joint_names()[1:], q_hist).replay(is_floating_base=False)