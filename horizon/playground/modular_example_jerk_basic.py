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

k = 2  # 2
ns = k*50  # number of shooting intervals

prb = problem.Problem(ns)
tf = 15.0  # [s]
dt = tf/ns

q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)
qddot = prb.createStateVariable("qddot", nv)

FD = kindyn.aba()
ID = cs.Function.deserialize(kindyn.rnea())

# jerk as input
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

q.setBounds(lb=q_min, ub=q_max)
qdot.setBounds(lb=(-qdot_lims).tolist(), ub=qdot_lims.tolist())

# set initial state
q0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# q0 = np.array([0, 0, 0, 0, 0, 0])

q.setBounds(lb=q0, ub=q0, nodes=0)
qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)
qddot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=0)

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
qinit = np.array([0, 0, 0, 0, 0, 0, 0, 0])
p_start = FK(q=qinit)['ee_pos'].toarray()

p_tgt = p_start.copy()
p_tgt[0] = 1.2
p_tgt[1] = 0.2
p_tgt[2] = 0.0

v = DFK(q=q, qdot=qdot)['ee_vel_linear']

homing_node = 20
prb.createConstraint('ee_tgt_1', p - p_tgt, nodes=k*homing_node)

quat = FK(q=q)['ee_rot']    

quat_start = FK(q=qinit)['ee_rot'].toarray()
quat_tgt = quat_start.copy()

z_versor = quat[:,2]
plane_normal = np.array([0,0,1])
dot_product = cs.dot(plane_normal, z_versor)

# the z of the end effector frame should point towards the plane. With the bounds we can relax this a bit
prb.createConstraint('orientation_constraint_1', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*homing_node)

qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*homing_node)

# we do the same for nodes 2
p_tgt[2] += 0.3
prb.createConstraint('ee_tgt_2', p - p_tgt, nodes=k*(5+homing_node))
prb.createConstraint('orientation_constraint_2', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(5+homing_node))

qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(5+homing_node))

# we do the same for nodes 3
p_tgt[1] -= 0.4
prb.createConstraint('ee_tgt_3', p - p_tgt, nodes=k*(10+homing_node))
prb.createConstraint('orientation_constraint_3', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(10+homing_node))

qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(10+homing_node))

# we do the same for nodes 4
p_tgt[2] -= 0.3
prb.createConstraint('ee_tgt_4', p - p_tgt, nodes=k*(15+homing_node))
prb.createConstraint('orientation_constraint_4', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(15+homing_node))

qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(15+homing_node))

# we do the same for nodes 5
p_tgt[2] += 0.3
prb.createConstraint('ee_tgt_5', p - p_tgt, nodes=k*(20+homing_node))
prb.createConstraint('orientation_constraint_5', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(20+homing_node))

qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(20+homing_node))

# we do the same for nodes 6
p_tgt[1] += 0.4
prb.createConstraint('ee_tgt_6', p - p_tgt, nodes=k*(25+homing_node))
prb.createConstraint('orientation_constraint_6', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(25+homing_node))

qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(25+homing_node))

# we do the same for nodes 7
p_tgt[2] -= 0.3
prb.createConstraint('ee_tgt_7', p - p_tgt, nodes=k*(30+homing_node))
prb.createConstraint('orientation_constraint_7', dot_product + 1.0, bounds=dict(lb=np.full(1, 0.0), ub=np.full(1, 0.001)), nodes=k*(30+homing_node))

qdot.setBounds(lb=np.zeros(nv), ub=np.zeros(nv), nodes=k*(30+homing_node))

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
    
plot_all = False
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
    
    # hplt.plotFunctions(legend=False)
    hplt.plotFunction('v_max', show_bounds=True)
    # plt.axhline(y=v_max**2, color='k', linestyle='--')
    hplt.plotFunction('inverse_dynamics', show_bounds=False)
    
    plt.show()


q_hist = solution["q"]
qdot_hist = solution["qdot"]
qddot_hist = solution["qddot"]
qdddot_hist = solution["qdddot"]


replay_trajectory(tf/ns, kindyn.joint_names()[1:], q_hist).replay(is_floating_base=False)