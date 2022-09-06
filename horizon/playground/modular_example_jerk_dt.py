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
# pkgpath = rospack.get_path('ModularBot_5DOF')
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

k = 1  # 2
ns = k*50  # number of shooting intervals

prb = problem.Problem(ns)
dt = prb.createVariable("dt", 1, nodes=list(range(0, ns)))

dt_min = [0.001] #[s]
dt_max = [1.0] #[s]
dt_init = [dt_min]
dt.setBounds(dt_min, dt_max)
dt.setInitialGuess(dt_init)

q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)
qddot = prb.createStateVariable("qddot", nv)

xy0 = prb.createSingleVariable("xy0", 2)
x_min = 0.0
x_max = 1.2
y_min = 0.0
y_max = 0.6
xy0_lb = np.array([x_min, y_min])
xy0_ub = np.array([x_max, y_max])
xy0.setBounds(lb=xy0_lb, ub=xy0_ub)

# q_init = np.array([0, 0, 0, 0, 0, 0, 0, 0])
q_init = cs.vertcat(xy0, cs.SX.zeros(nq-2,1))

FD = kindyn.aba()
ID = cs.Function.deserialize(kindyn.rnea())

# jerk as input
qdddot = prb.createInputVariable("qdddot", nv)
        
x = prb.getState().getVars()
u = prb.getInput().getVars()

# specify derivative of state vector
xdot = cs.vertcat(qdot, qddot, qdddot)
prb.setDynamics(xdot)
prb.setDt(dt)

# transcription
th = Transcriptor.make_method(trans, prb, opts=dict(integrator='RK4'))

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

q.setBounds(lb=q_min, ub=q_max)
qdot.setBounds(lb=(-qdot_lims).tolist(), ub=qdot_lims.tolist())

# set initial state
# q0_lb = np.array([x_min, y_min, 0, 0, 0, 0, 0, 0])
q0_lb = np.concatenate((xy0_lb, np.full(nq-2, 0.0)))  #, axis=1)
# q0_ub = np.array([x_max, y_max, 0, 0, 0, 0, 0, 0])
q0_ub = np.concatenate((xy0_ub, np.full(nq-2, 0.0)))  #, axis=1)

q.setBounds(lb=q0_lb, ub=q0_ub, nodes=0)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)

# Optimize for base placement
# constraint on intital position: this is to optimize the base location!
prb.createConstraint('initial_position', q-q_init, nodes=0)
# and the first two joint should not move:
prb.createConstraint('base_zero_velocity', qdot[0:2])

# zero final velocity 
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=ns)
# prb.createFinalConstraint('qdot_final', qdot)

# zero final acceleration
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=ns)
# prb.createFinalConstraint('qddot_final', qddot)

# # zero initial jerk
# qdddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)
# # zero final jerk
# qdddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=ns-1)

# add inverse dynamics as constraint
tau = ID(q=q, v=qdot, a=qddot)['tau']       
tau_cnstrnt = prb.createIntermediateConstraint("inverse_dynamics", tau, bounds=dict(lb=-tau_lims, ub=tau_lims))

# cartesian target
frame = 'pen_A'
FK = cs.Function.deserialize(kindyn.fk(frame))
DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

p = FK(q=q)['ee_pos']
p_start = FK(q=q_init)['ee_pos']  # .toarray()

p_tgt = np.array([0.0,0.0,0.0])  # p_start.copy()
p_tgt[0] = 1.2
p_tgt[1] = 0.2
p_tgt[2] = 0.0

v = DFK(q=q, qdot=qdot)['ee_vel_linear']

quat = FK(q=q)['ee_rot']    

quat_start = FK(q=q_init)['ee_rot']  # .toarray()
quat_tgt = np.array([0.0,0.0,0.0])  # quat_start.copy()

z_versor = quat[:,2]
plane_normal = np.array([0,0,1])
dot_product = cs.dot(plane_normal, z_versor)

# knot 1
homing_node = k*20
node_list_1 = range(homing_node, homing_node + 2)
prb.createConstraint('ee_tgt_1', p - p_tgt, nodes=node_list_1)

# the z of the end effector frame should point towards the plane. With the bounds we can relax this a bit
prb.createConstraint('orientation_constraint_1', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=node_list_1)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node_list_1)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node_list_1)
dt.setBounds(1.0, 1.0, nodes = homing_node)

# we do the same for knot 2
node2 = k*5+homing_node
node_list_2 = range(node2, node2 + 2) 
p_tgt[2] += 0.3
prb.createConstraint('ee_tgt_2', p - p_tgt, nodes=node2)
prb.createConstraint('orientation_constraint_2', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=node2)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node2)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node2)
#dt.setBounds(1.0, 1.0, nodes = node2)

# we do the same for knot 3
node3 = k*10+homing_node
node_list_3 = range(node3, node3 + 2) 
p_tgt[1] -= 0.4
prb.createConstraint('ee_tgt_3', p - p_tgt, nodes=node3)
prb.createConstraint('orientation_constraint_3', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=node3)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node3)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node3)
# dt.setBounds(1.0, 1.0, nodes = node3)

# we do the same for knot 4
node4 = k*15+homing_node
node_list_4 = range(node4, node4 + 2) 
p_tgt[2] -= 0.3
prb.createConstraint('ee_tgt_4', p - p_tgt, nodes=node_list_4)
prb.createConstraint('orientation_constraint_4', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=node_list_4)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node_list_4)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node_list_4)
dt.setBounds(1.0, 1.0, nodes = node4)

# we do the same for knot 5
node5 = k*20+homing_node
node_list_5 = range(node5, node5 + 2) 
p_tgt[2] += 0.3
prb.createConstraint('ee_tgt_5', p - p_tgt, nodes=node5)
prb.createConstraint('orientation_constraint_5', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=node5)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node5)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node5)
# dt.setBounds(1.0, 1.0, nodes = node5)

# we do the same for knot 6
node6 = k*25+homing_node
node_list_6 = range(node6, node6 + 2) 
p_tgt[1] += 0.4
prb.createConstraint('ee_tgt_6', p - p_tgt, nodes=node6)
prb.createConstraint('orientation_constraint_6', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=node6)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node6)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node6)
# dt.setBounds(1.0, 1.0, nodes = node6)

# we do the same for knot 7
node7=k*30+homing_node
node_list_7 = range(node7, node7 + 2) 
p_tgt[2] -= 0.3
prb.createConstraint('ee_tgt_7', p - p_tgt, nodes=node7)
prb.createConstraint('orientation_constraint_7', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=node7)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node7)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=node7)
# dt.setBounds(1.0, 1.0, nodes = node7)

# Compute moving mass in an ugly way :(
moving_mass = 0.0
moving_links = { k: v for k, v in robot.link_map.items()\
     if k not in ['world', 'base_link', 'J1_A_stator', 'pen_A', 'virtual_link'] }
for link_name, link_object in moving_links.items():
    if link_object.inertial is not None:
        moving_mass += link_object.inertial.mass
# From ISO/TS 15066
m_R = moving_mass / 2.0

# Compute max safe velocity
v_max = compute_vel_max(m_R, body_part='chest', formula='use_energy')

# Limit on max velocity
prb.createConstraint("v_max", cs.sumsqr(v), bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, v_max**2)))
#prb.createConstraint("v_max", cs.sumsqr(v) - v_max**2, bounds=dict(lb=np.full(1, -cs.inf), ub=np.full(1, 0.0)))

##### Cost function #####
# Cost
prb.createIntermediateCost('min_f', 10000 * cs.sum2(dt))  # nodes=range(homing_node,ns+1)
# penalize input
prb.createIntermediateCost('min_u', 1 * cs.sumsqr(qdddot))

# create solver
prb_solver = solver.Solver.make_solver(
    solver_type, 
    prb,
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

tf = np.sum(solution["dt"])
print(f"Tf: {tf}")
cycle_tyme = np.sum(solution["dt"][0,homing_node:])
print(f"Cycle Time: {cycle_tyme}")

xy0 = solution["xy0"]
print(f"xy0: {xy0}")

print(f"v_max: {v_max}")
print(f"m_R: {m_R}")

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

    hplt.plotVariable('dt', grid=False, show_bounds=True)

    hplt.plotVariable('xy0', grid=True, markers='x', show_bounds=True)
    
    # hplt.plotFunctions(legend=False)
    hplt.plotFunction('v_max', show_bounds=True)
    # plt.axhline(y=v_max**2, color='k', linestyle='--')
    hplt.plotFunction('inverse_dynamics', show_bounds=True)
    
    hplt.plotFunction('base_zero_velocity', show_bounds=True)

    plt.show()


q_hist = solution["q"]
qdot_hist = solution["qdot"]
qddot_hist = solution["qddot"]
qdddot_hist = solution["qdddot"]


#resample trajectory
L = 0.5*cs.dot(qdot, qdot)  # Objective term
dae = {'x': x, 'p': qdddot, 'ode': xdot, 'quad': L}

dt = 0.01
dt_hist = solution["dt"]
dt_old = dt_hist.flatten()

state_res = resampler_trajectory.resampler(np.array(cs.vertcat(q_hist, qdot_hist, qddot_hist)), qdddot_hist, dt_old, dt, dae)
q_res = state_res[0:nq, :]
qdot_res = state_res[nq:2*nq, :]
qddot_res = state_res[2*nq:3*nq, :]

qdddot_res = resampler_trajectory.resample_input(qdddot_hist, dt_old, dt)

PLOTS = False
if PLOTS:
    time = np.arange(0.0, state_res.shape[1]*dt, dt)
    # this must be a bug on the resampler: HACK
    if nq == 7:
        time = time[:-1]    

    plt.figure()
    for i in range(2):
        plt.plot(time, q_res[i, :])
    plt.suptitle('$\mathrm{x_0}, \mathrm{y_0}$', size=20)
    plt.xlabel('$\mathrm{[sec]}$', size=20)
    plt.ylabel('$\mathrm{[m]}$', size=20)

    plt.figure()
    for i in range(2,nq):
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