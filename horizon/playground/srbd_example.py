#!/usr/bin/env python
import casadi
import logging

import rospy
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.transcriptions import integrators
from horizon.solvers import solver
from horizon.ros.replay_trajectory import *
import matplotlib.pyplot as plt
import os
import time
from horizon.ros import utils as horizon_ros_utils
from ttictoc import tic,toc
import tf
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy
from numpy import linalg as LA
from visualization_msgs.msg import Marker, MarkerArray
from abc import ABCMeta, abstractmethod

# Create horizon problem
ns = 20
prb = problem.Problem(ns)
T = 1.


class action_class(metaclass = ABCMeta):
    @abstractmethod
    def action(self, nodes):
        pass

class action_D(action_class):
    def __init__(self, f, cdot):
        self.f = f
        self.cdot = cdot

    def action(self, nodes):
        for key, var in self.f.items():
            var.setBounds([-1000., -1000., -1000.], [1000., 1000., 1000.], nodes=nodes)
        for key, var in self.cdot.items():
            var.setBounds([0., 0., 0.], [0., 0., 0.], nodes=nodes)

class action_L(action_class):
    def __init__(self, f, cdot):
        self.f = f
        self.cdot = cdot

    def action(self, nodes):
        for i in range(0, 4):
            self.f[i].setBounds([-1000., -1000., -1000.], [1000., 1000., 1000.], nodes=nodes)
            self.cdot[i].setBounds([0., 0., 0.], [0., 0., 0.], nodes=nodes)
        for i in range(4, 8):
            self.f[i].setBounds([0., 0., 0.], [0., 0., 0.], nodes=nodes)
            self.cdot[i].setBounds([-10., -10., -10.], [10., 10., 10.], nodes=nodes)

class action_R(action_class):
    def __init__(self, f, cdot):
        self.f = f
        self.cdot = cdot

    def action(self, nodes):
        for i in range(4, 8):
            self.f[i].setBounds([-1000., -1000., -1000.], [1000., 1000., 1000.], nodes=nodes)
            self.cdot[i].setBounds([0., 0., 0.], [0., 0., 0.], nodes=nodes)
        for i in range(0, 4):
            self.f[i].setBounds([0., 0., 0.], [0., 0., 0.], nodes=nodes)
            self.cdot[i].setBounds([-10., -10., -10.], [10., 10., 10.], nodes=nodes)

class action_scheduler:
    def __init__(self, action_dict, batch_dict):
        self._action_dict = action_dict
        self._batch_dict = batch_dict

        self._batch = list()
        self._initial_batch = list()
        self._next_batch = list()

    def setInitialBatch(self, key):
        self._batch = self._batch_dict[key].copy()
        self._initial_batch = self._batch_dict[key].copy()

    def setNextBatch(self, key):
        if not self._next_batch:
            self._next_batch = self._batch_dict[key].copy()

    def execute(self):
        n = 0
        for action in self._batch:
            self._action_dict[action].action(n)
            n = n + 1

        if not self._next_batch:
            self._next_batch = self._initial_batch.copy()

        self._batch.pop(0)
        self._batch.append(self._next_batch.pop(0))



def joy_cb(msg):
    global joy_msg
    joy_msg = msg

def publishContactForce(t, f, frame):

    f_msg = WrenchStamped()
    f_msg.header.stamp = t
    f_msg.header.frame_id = frame

    f_msg.wrench.force.x = f[0]
    f_msg.wrench.force.y = f[1]
    f_msg.wrench.force.z = f[2]

    f_msg.wrench.torque.x = 0.
    f_msg.wrench.torque.y = 0.
    f_msg.wrench.torque.z = 0.

    pub = rospy.Publisher('f' + frame, WrenchStamped, queue_size=10).publish(f_msg)



def SRBDTfBroadcaster(r, o, c_dict, t):
    br = tf.TransformBroadcaster()

    br.sendTransform(r,o,t,"SRB","world")
    for key, val in c_dict.items():
        br.sendTransform(val, [0., 0., 0., 1.], t, key, "world")

def SRBDViewer(I, base_frame, t, number_of_contacts):
    marker = Marker()
    marker.header.frame_id = base_frame
    marker.header.stamp = t
    marker.ns = "SRBD"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = marker.pose.position.y = marker.pose.position.z = 0.
    marker.pose.orientation.x = marker.pose.orientation.y = marker.pose.orientation.z = 0.
    marker.pose.orientation.w = 1.

    a = I[0,0] + I[1,1] + I[2,2]
    marker.scale.x = 0.5*(I[2,2] + I[1,1])/a
    marker.scale.y = 0.5*(I[2,2] + I[0,0])/a
    marker.scale.z = 0.5*(I[0,0] + I[1,1])/a
    marker.color.a = 0.8
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0

    pub = rospy.Publisher('box', Marker, queue_size=10).publish(marker)

    marker_array = MarkerArray()
    for i in range(0, number_of_contacts):
        m = Marker()
        m.header.frame_id = "c" + str(i)
        m.header.stamp = t
        m.ns = "SRBD"
        m.id = i + 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = marker.pose.position.y = marker.pose.position.z = 0.
        m.pose.orientation.x = marker.pose.orientation.y = marker.pose.orientation.z = 0.
        m.pose.orientation.w = 1.

        m.scale.x = 0.04
        m.scale.y = 0.04
        m.scale.z = 0.04
        m.color.a = 0.8
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0

        marker_array.markers.append(m)

    pub2 = rospy.Publisher('contacts', MarkerArray, queue_size=10).publish(marker_array)



horizon_ros_utils.roslaunch("horizon_examples", "SRBD_kangaroo.launch")
time.sleep(3.)

urdf = rospy.get_param("robot_description", "")
if urdf == "":
    print("robot_description not loaded in param server!")
    exit()

kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# Creates problem STATE variables
r = prb.createStateVariable("r", 3) # CoM position
r_prev = r.getVarOffset(-1)

rdot = prb.createStateVariable("rdot", 3) # CoM vel
rdot_prev = rdot.getVarOffset(-1)
rdot_ref = prb.createParameter('rdot_ref', 3)
rdot_ref.assign([0. ,0. , 0.], nodes=range(1, ns+1))

rddot = prb.createInputVariable("rddot", 3) # CoM acc
rddot_prev = rddot.getVarOffset(-1)

o = prb.createStateVariable("o", 4) # base orientation
o_prev = o.getVarOffset(-1)
w = prb.createStateVariable("w", 3) # base vel
w_prev = w.getVarOffset(-1)
w_ref = prb.createParameter('w_ref', 3)
w_ref.assign([0. ,0. , 0.], nodes=range(1, ns+1))

wdot = prb.createInputVariable("wdot", 3) # base acc
wdot_prev = wdot.getVarOffset(-1)

q = cs.vertcat(r, o)
qdot = cs.vertcat(rdot, w)
qddot = cs.vertcat(rddot, wdot)

q_prev = cs.vertcat(r_prev, o_prev)
qdot_prev = cs.vertcat(rdot_prev, w_prev)
qddot_prev = cs.vertcat(rddot_prev, wdot_prev)

nc = 2 * 4
c = dict()
cdot = dict()
cddot = dict()
f = dict()
for i in range(0, nc):
    c[i] = prb.createStateVariable("c" + str(i), 3) # Contact i position
    q = cs.vertcat(q, c[i])
    q_prev = cs.vertcat(q_prev, c[i].getVarOffset(-1))

    cdot[i] = prb.createStateVariable("cdot" + str(i), 3)  # Contact i vel
    qdot = cs.vertcat(qdot, cdot[i])
    qdot_prev = cs.vertcat(qdot_prev, cdot[i].getVarOffset(-1))

    cddot[i] = prb.createInputVariable("cddot" + str(i), 3) # Contact i acc
    qddot = cs.vertcat(qddot, cddot[i])
    qddot_prev = cs.vertcat(qddot_prev, cddot[i].getVarOffset(-1))

    f[i] = prb.createInputVariable("f" + str(i), 3) # Contact i forces



print(f"q : {q}")
print(f"qdot : {qdot}")
print(f"qddot : {qddot}")

# Formulate discrete time dynamics
x, xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)
prb.setDynamics(xdot)
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': 0}
F_integrator = integrators.RK2(dae, opts=None)

#Limits
joint_init = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  # px, py, pz, qx, qy, qz, qw
               0.0, 0.04999999999997918, 0.0,  # 'leg_left_1_joint', 'leg_left_2_joint', 'leg_left_3_joint',
               1.1693870096573384, 2.3401777720374923, 0.6999999999999997, # 'leg_left_femur_joint', 'leg_left_knee_joint', 'leg_left_length_joint'
              -0.049999999999970346, 0.0,  # 'leg_left_4_joint', 'leg_left_5_joint'
               0.0, 0.04999999999997918, 0.0,  # 'leg_right_1_joint', 'leg_right_2_joint', 'leg_right_3_joint',
               1.1693870096573384, 2.3401777720374923, 0.6999999999999997, # 'leg_right_femur_joint', 'leg_right_knee_joint', 'leg_right_length_joint'
              -0.049999999999970346, 0.0]  # 'leg_right_4_joint', 'leg_right_5_joint'
foot_frames = ["left_foot_upper_left", "left_foot_upper_right", "left_foot_lower_left",
                                 "left_foot_lower_right", # contact 0 to 3
                "right_foot_upper_left", "right_foot_upper_right", "right_foot_lower_left",
                                 "right_foot_lower_right"] # contact 4 to 7

i = 0
initial_foot_position = dict()
for frame in foot_frames:
    FK = cs.Function.deserialize(kindyn.fk(frame))
    p = FK(q=joint_init)['ee_pos']
    print(f"{frame}: {p}")
    initial_foot_position[i] = p

    c[i].setInitialGuess(p)
    c[i].setBounds(p, p, 0)  # starts in homing

    cdot[i].setInitialGuess([0., 0., 0.])
    cdot[i].setBounds([0., 0., 0.], [0., 0., 0.], 0)  # starts with 0 velocity
    cdot[i].setBounds([0., 0., 0.], [0., 0., 0.], ns)  # ends with 0 velocity

    f[i].setBounds([-1000., -1000., -1000.], [1000., 1000., 1000.])

    i = i + 1

COM = cs.Function.deserialize(kindyn.centerOfMass())
com = COM(q=joint_init)['com']
print(f"com: {com}")
r.setInitialGuess(com)
r.setBounds(com, com, 0)
rdot.setInitialGuess([0., 0., 0.])
#rdot.setBounds([0., 0., 0.], [0., 0., 0.], 0)

print(f"base orientation: {joint_init[3:7]}")
o.setInitialGuess(joint_init[3:7])
o.setBounds(joint_init[3:7], joint_init[3:7], 0)
w.setInitialGuess([0., 0., 0.])
w.setBounds([0., 0., 0.], [0., 0., 0.], 0)

# SET UP COST FUNCTION
prb.createCost("rz_tracking", 2e3 * cs.sumsqr(r[2] - com[2]), nodes=range(1, ns+1))
Wo = prb.createParameter('Wo', 1)
Wo.assign(0.)
prb.createCost("o_tracking", Wo * cs.sumsqr(o - joint_init[3:7]), nodes=range(1, ns+1))
prb.createCost("rdot_tracking", 1e6 * cs.sumsqr(rdot - rdot_ref), nodes=range(1, ns+1))
prb.createCost("w_tracking", 1e6*cs.sumsqr(w - w_ref), nodes=range(1, ns+1))
prb.createCost("min_qddot", 1e1*cs.sumsqr(qddot), nodes=list(range(0, ns)))

for i in range(0, 4):
    prb.createCost("min_f" + str(i), 1e-3 * cs.sumsqr(f[i]), nodes=list(range(0, ns)))
    prb.createCost("min_cdoty" + str(i), 1e6 * cs.sumsqr(cdot[i][1]))
    prb.createCost("min_cdotz" + str(i), 1e6 * cs.sumsqr(cdot[i][2]))
for i in range(4, 8):
    prb.createCost("min_f" + str(i), 1e-3 * cs.sumsqr(f[i]), nodes=list(range(0, ns)))
    prb.createCost("min_cdoty" + str(i), 1e6 * cs.sumsqr(cdot[i][1]))
    prb.createCost("min_cdotz" + str(i), 1e6 * cs.sumsqr(cdot[i][2]))


# CONSTRAINTS
x_prev, _ = utils.double_integrator_with_floating_base(q_prev, qdot_prev, qddot_prev)
x_int = F_integrator(x0=x_prev, p=qddot_prev, time=T/ns)
prb.setDt(T/ns)
prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=list(range(1, ns+1)), bounds=dict(lb=np.zeros(x.size1()), ub=np.zeros(x.size1())))

# FEET
for i, fi in f.items():
    # FRICTION CONE
    mu = 0.8  # friction coefficient
    R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(fi, mu, R)
    prb.createIntermediateConstraint(f"f{i}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))

prb.createConstraint("relative_vel_left_1", cdot[0] - cdot[1])
prb.createConstraint("relative_vel_left_2", cdot[0] - cdot[2])
prb.createConstraint("relative_vel_left_3", cdot[0] - cdot[3])
prb.createConstraint("relative_vel_right_1", cdot[4] - cdot[5])
prb.createConstraint("relative_vel_right_2", cdot[4] - cdot[6])
prb.createConstraint("relative_vel_right_3", cdot[4] - cdot[7])

m = kindyn.mass()
print(f"mass: {m}")
M = cs.Function.deserialize(kindyn.crba())
I = M(q=joint_init)['B'][3:6, 3:6]
print(f"I centroidal: {I}")

SRBD = kin_dyn.SRBD(m, I, f, r, rddot, c, w, wdot)
prb.createConstraint("SRBD", SRBD, bounds=dict(lb=np.zeros(6), ub=np.zeros(6)), nodes=list(range(0, ns)))

# Create problem
opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 5000,
        'ipopt.linear_solver': 'ma57',
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'print_time': 0,
        'print_level': 0}

solver = solver.Solver.make_solver('ipopt', prb, opts)

tic()
solver.solve()
print(f"time first solve: {toc()}")

solution = solver.getSolutionDict()

variables_dict = {"r": r, "rdot": rdot, "rddot": rddot,
                  "o": o, "w": w, "wdot": wdot}
for i in range(0, 8):
    variables_dict["c" + str(i)] = c[i]
    variables_dict["cdot" + str(i)] = cdot[i]
    variables_dict["cddot" + str(i)] = cddot[i]
    variables_dict["f" + str(i)] = f[i]

rospy.init_node('srbd_mpc_test', anonymous=True)
rate = rospy.Rate(10)  # 10hz
rospy.Subscriber('/joy', Joy, joy_cb)
global joy_msg
joy_msg = rospy.wait_for_message("joy", Joy)


D = action_D(f, cdot)
L = action_L(f, cdot)
R = action_R(f, cdot)
actions = {"D": D, "L": L, "R": R}

REST = list()
for i in range(0, ns):
    REST.append("D")

STEP = list()
for i in range(0, 3):
    STEP.append("D")
for i in range(3, 10):
    STEP.append("L")
for i in range(10, 13):
    STEP.append("D")
for i in range(13, ns):
    STEP.append("R")

batch = {"REST": REST, "STEP": STEP}

scheduler = action_scheduler(actions, batch)
scheduler.setInitialBatch("REST")

while not rospy.is_shutdown():
    mat_storer.setInitialGuess(variables_dict, solution)
    #open loop
    r.setBounds(solution['r'][:, 1], solution['r'][:, 1], 0)
    rdot.setBounds(solution['rdot'][:, 1], solution['rdot'][:, 1], 0)
    o.setBounds(solution['o'][:, 1], solution['o'][:, 1], 0)
    w.setBounds(solution['w'][:, 1], solution['w'][:, 1], 0)
    for i in range(0, 8):
        c[i].setBounds(solution['c' + str(i)][: ,1], solution['c' + str(i)][: ,1], 0)
        cdot[i].setBounds(solution['cdot' + str(i)][:, 1], solution['cdot' + str(i)][:, 1], 0)


    #JOYSTICK
    rdot_ref.assign([0.1 * joy_msg.axes[1], -0.1 * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=range(1, ns+1)) #com velocities
    w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=range(1, ns + 1)) #base angular velocities

    if(joy_msg.buttons[3]):
        Wo.assign(1e5)
    else:
        Wo.assign(0.)


    scheduler.execute()

    if joy_msg.buttons[4]:
        scheduler.setNextBatch("STEP")
        print("STEP")
    #else:
    #    setForceLimits(0,4, f, False)

    ##


    tic()
    if not solver.solve():
        print("UNABLE TO SOLVE")
    solution = solver.getSolutionDict()
    print(f"time solve: {toc()}")

    c0_hist = dict()
    for i in range(0, 8):
        c0_hist['c' + str(i)] = solution['c' + str(i)][:, 0]

    t = rospy.Time().now()
    SRBDTfBroadcaster(solution['r'][:, 0], solution['o'][:, 0], c0_hist, t)
    for i in range(0, 8):
        publishContactForce(t, solution['f' + str(i)][:, 0], 'c' + str(i))
    SRBDViewer(I, "SRB", t, 8)








