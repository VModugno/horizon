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

class steps_phase:
    def __init__(self, f, c, cdot, c_init_z, c_ref, nodes):
        self.f = f
        self.c = c
        self.cdot = cdot
        self.c_ref = c_ref

        self.nodes = nodes
        self.step_counter = 0

        #JUMP
        self.l_jump = []
        self.l_jump_cdot_bounds = []
        self.l_jump_f_bounds = []
        self.r_jump = []
        self.r_jump_cdot_bounds = []
        self.r_jump_f_bounds = []
        sin = 0.1 * np.sin(np.linspace(0, np.pi, 8))
        for k in range(0, 7):  # 7 nodes down
            self.l_jump.append(c_init_z)
            self.r_jump.append(c_init_z)
            self.l_jump_cdot_bounds.append([0., 0., 0.])
            self.r_jump_cdot_bounds.append([0., 0., 0.])
            self.l_jump_f_bounds.append([1000., 1000., 1000.])
            self.r_jump_f_bounds.append([1000., 1000., 1000.])
        for k in range(0, 8):  # 8 nodes jump
            self.l_jump.append(c_init_z + sin[k])
            self.r_jump.append(c_init_z + sin[k])
            self.l_jump_cdot_bounds.append([10., 10., 10.])
            self.r_jump_cdot_bounds.append([10., 10., 10.])
            self.l_jump_f_bounds.append([0., 0., 0.])
            self.r_jump_f_bounds.append([0., 0., 0.])
        for k in range(0, 7):  # 6 nodes down
            self.l_jump.append(c_init_z)
            self.r_jump.append(c_init_z)
            self.l_jump_cdot_bounds.append([0., 0., 0.])
            self.r_jump_cdot_bounds.append([0., 0., 0.])
            self.l_jump_f_bounds.append([1000., 1000., 1000.])
            self.r_jump_f_bounds.append([1000., 1000., 1000.])



        #NO STEP
        self.stance = []
        self.cdot_bounds = []
        self.f_bounds = []
        for k in range(0, nodes):
            self.stance.append([c_init_z])
            self.cdot_bounds.append([0., 0., 0.])
            self.f_bounds.append([1000., 1000., 1000.])


        #STEP
        sin = 0.1 * np.sin(np.linspace(0, np.pi, 8))
        #left step cycle
        self.l_cycle = []
        self.l_cdot_bounds = []
        self.l_f_bounds = []
        for k in range(0,2): # 2 nodes down
            self.l_cycle.append(c_init_z)
            self.l_cdot_bounds.append([0., 0., 0.])
            self.l_f_bounds.append([1000., 1000., 1000.])
        for k in range(0, 8):  # 8 nodes step
            self.l_cycle.append(c_init_z + sin[k])
            self.l_cdot_bounds.append([10., 10., 10.])
            self.l_f_bounds.append([0., 0., 0.])
        for k in range(0, 2):  # 2 nodes down
            self.l_cycle.append(c_init_z)
            self.l_cdot_bounds.append([0., 0., 0.])
            self.l_f_bounds.append([1000., 1000., 1000.])
        for k in range(0, 8):  # 8 nodes down (other step)
            self.l_cycle.append(c_init_z)
            self.l_cdot_bounds.append([0., 0., 0.])
            self.l_f_bounds.append([1000., 1000., 1000.])
        self.l_cycle.append(c_init_z) # last node down
        self.l_cdot_bounds.append([0., 0., 0.])
        self.l_f_bounds.append([1000., 1000., 1000.])

        # right step cycle
        self.r_cycle = []
        self.r_cdot_bounds = []
        self.r_f_bounds = []
        for k in range(0, 2):  # 2 nodes down
            self.r_cycle.append(c_init_z)
            self.r_cdot_bounds.append([0., 0., 0.])
            self.r_f_bounds.append([1000., 1000., 1000.])
        for k in range(0, 8):  # 8 nodes down (other step)
            self.r_cycle.append(c_init_z)
            self.r_cdot_bounds.append([0., 0., 0.])
            self.r_f_bounds.append([1000., 1000., 1000.])
        for k in range(0, 2):  # 2 nodes down
            self.r_cycle.append(c_init_z)
            self.r_cdot_bounds.append([0., 0., 0.])
            self.r_f_bounds.append([1000., 1000., 1000.])
        for k in range(0, 8):  # 8 nodes step
            self.r_cycle.append(c_init_z + sin[k])
            self.r_cdot_bounds.append([10., 10., 10.])
            self.r_f_bounds.append([0., 0., 0.])
        self.r_cycle.append(c_init_z)  # last node down
        self.r_cdot_bounds.append([0., 0., 0.])
        self.r_f_bounds.append([1000., 1000., 1000.])

        self.action = ""

    def set(self, action):
        t = self.nodes - self.step_counter # this goes FROM nodes TO 0

        for k in range(max(t, 0), self.nodes + 1):
            ref_id = (k - t)%self.nodes

            if(ref_id == 0):
                self.action = action

            if action == "step":
                for i in range(0, 4):
                    self.c_ref[i].assign(self.l_cycle[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.l_cdot_bounds[ref_id]), np.array(self.l_cdot_bounds[ref_id]), nodes=k)
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.l_f_bounds[ref_id]), np.array(self.l_f_bounds[ref_id]), nodes=k)
                for i in range(4, 8):
                    self.c_ref[i].assign(self.r_cycle[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.r_cdot_bounds[ref_id]), np.array(self.r_cdot_bounds[ref_id]), nodes=k)
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.r_f_bounds[ref_id]), np.array(self.r_f_bounds[ref_id]), nodes=k)

            elif action == "jump":
                for i in range(0, 4):
                    self.c_ref[i].assign(self.l_jump[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.l_jump_cdot_bounds[ref_id]), np.array(self.l_jump_cdot_bounds[ref_id]), nodes=k)
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.l_jump_f_bounds[ref_id]), np.array(self.l_jump_f_bounds[ref_id]), nodes=k)
                for i in range(4, 8):
                    self.c_ref[i].assign(self.r_jump[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.r_jump_cdot_bounds[ref_id]), np.array(self.r_jump_cdot_bounds[ref_id]), nodes=k)
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.r_jump_f_bounds[ref_id]), np.array(self.r_jump_f_bounds[ref_id]), nodes=k)

            else:
                for i in range(0, 8):
                    self.c_ref[i].assign(self.stance[ref_id], nodes=k)
                    self.cdot[i].setBounds(-1. * np.array(self.cdot_bounds[ref_id]),
                                           np.array(self.cdot_bounds[ref_id]), nodes=k)
                    if k < self.nodes:
                        self.f[i].setBounds(-1. * np.array(self.f_bounds[ref_id]), np.array(self.f_bounds[ref_id]),
                                        nodes=k)

        self.step_counter += 1



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
    cdot[i].setBounds([0., 0., 0.], [0., 0., 0.])  # with 0 velocity

    f[i].setBounds([-1000., -1000., -1000.], [1000., 1000., 1000.])

    i = i + 1

COM = cs.Function.deserialize(kindyn.centerOfMass())
com = COM(q=joint_init)['com']
print(f"com: {com}")
r.setInitialGuess(com)
r.setBounds(com, com, 0)
rdot.setInitialGuess([0., 0., 0.])

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

prb.createCost("rdot_tracking", 1e4 * cs.sumsqr(rdot - rdot_ref), nodes=range(1, ns+1))

prb.createCost("w_tracking", 1e6*cs.sumsqr(w - w_ref), nodes=range(1, ns+1))

prb.createCost("min_qddot", 1e0*cs.sumsqr(qddot), nodes=list(range(0, ns)))

#these are the relative distance in y between the feet (needs to be rotated!)
d_initial_1 = -(initial_foot_position[1][1] - initial_foot_position[4][1])
relative_pos_y_1_4 = prb.createConstraint("relative_pos_y_1_4", -c[1][1] + c[4][1], bounds=dict(ub= d_initial_1, lb=d_initial_1-0.5))
d_initial_2 = -(initial_foot_position[3][1] - initial_foot_position[6][1])
relative_pos_y_3_6 = prb.createConstraint("relative_pos_y_3_6", -c[3][1] + c[6][1], bounds=dict(ub= d_initial_2, lb=d_initial_2-0.5))


c_ref = dict()
for i in range(0, 4):
    prb.createCost("min_f" + str(i), 1e-3 * cs.sumsqr(f[i]), nodes=list(range(0, ns)))

    c_ref[i] = prb.createParameter("c_ref" + str(i), 1)
    c_ref[i].assign(initial_foot_position[i][2], nodes=range(0, ns+1))
    prb.createConstraint("min_cz" + str(i), c[i][2] - c_ref[i])

for i in range(4, 8):
    prb.createCost("min_f" + str(i), 1e-3 * cs.sumsqr(f[i]), nodes=list(range(0, ns)))

    c_ref[i] = prb.createParameter("c_ref" + str(i), 1)
    c_ref[i].assign(initial_foot_position[i][2], nodes=range(0, ns+1))
    prb.createConstraint("min_cz" + str(i), c[i][2] - c_ref[i])


# CONSTRAINTS
x_prev, _ = utils.double_integrator_with_floating_base(q_prev, qdot_prev, qddot_prev)
x_int = F_integrator(x0=x_prev, p=qddot_prev, time=T/ns)
prb.setDt(T/ns)
prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=list(range(1, ns+1)), bounds=dict(lb=np.zeros(x.size1()), ub=np.zeros(x.size1())))

# FEET
for i, fi in f.items():
    # FRICTION CONE
    mu = 0.8  # friction coefficient
    StanceR = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(fi, mu, StanceR)
    prb.createIntermediateConstraint(f"f{i}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))

#these are to keep the 4 points as a feet (needs to consider w x p)
prb.createConstraint("relative_vel_left_1", cdot[0][0:2] - cdot[1][0:2])
prb.createConstraint("relative_vel_left_2", cdot[0][0:2] - cdot[2][0:2])
prb.createConstraint("relative_vel_left_3", cdot[0][0:2] - cdot[3][0:2])

prb.createConstraint("relative_vel_right_1", cdot[4][0:2] - cdot[5][0:2])
prb.createConstraint("relative_vel_right_2", cdot[4][0:2] - cdot[6][0:2])
prb.createConstraint("relative_vel_right_3", cdot[4][0:2] - cdot[7][0:2])


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
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.fast_step_computation': 'yes',
        'ipopt.print_level': 0,
        'ipopt.sb': 'no',
        'print_time': False,
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


wpg = steps_phase(f, c, cdot, initial_foot_position[0][2].__float__(), c_ref, ns)
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
    alphaX = alphaY = 0.1
    if joy_msg.buttons[4] or joy_msg.buttons[5]:
        alphaX = 0.4
        alphaY = 0.3

    rdot_ref.assign([alphaX * joy_msg.axes[1], -alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=range(1, ns+1)) #com velocities
    w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=range(1, ns + 1)) #base angular velocities

    if(joy_msg.buttons[3]):
        Wo.assign(1e5)
    else:
        Wo.assign(0.)

    if(joy_msg.buttons[4]):
        wpg.set("step")
        relative_pos_y_1_4.setBounds(ub=d_initial_1, lb=d_initial_1 - 0.5)
        relative_pos_y_3_6.setBounds(ub=d_initial_2, lb=d_initial_2 - 0.5)
    elif (joy_msg.buttons[5]):
        wpg.set("jump")
        relative_pos_y_1_4.setBounds(ub=d_initial_1, lb=d_initial_1)
        relative_pos_y_3_6.setBounds(ub=d_initial_2, lb=d_initial_2)
    else:
        wpg.set("cazzi")
        relative_pos_y_1_4.setBounds(ub=d_initial_1, lb=d_initial_1 - 0.5)
        relative_pos_y_3_6.setBounds(ub=d_initial_2, lb=d_initial_2 - 0.5)




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








