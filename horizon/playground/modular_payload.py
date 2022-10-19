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

# options
solver_type = 'ipopt'
use_orientation_as_constraint = True
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

tf = 4.0
n_nodes = 100
dt = tf / n_nodes

# # load URDF
# rospack = rospkg.RosPack()
# pkgpath = rospack.get_path('ModularBot_michè')
# # pkgpath = rospack.get_path('ModularBot_5DOF')
# urdfpath = os.path.join(pkgpath, 'urdf', 'ModularBot.urdf')
# srdfpath = os.path.join(pkgpath, 'srdf', 'ModularBot.srdf')
# urdf = open(urdfpath, 'r').read()

def main():
    app.run(host='0.0.0.0', port=5050, debug=True, threaded=True)

def compute_payload(urdf, tip_link, is_online):
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
    nq = kindyn.nq()
    nv = kindyn.nv()

    # joint names
    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

    # define dynamics
    prb = problem.Problem(n_nodes, receding=False, abstract_casadi_type=cs.MX) #, casadi_type=cs.SX)
    # q = prb.createStateVariable('q', nq)
    # q_dot = prb.createStateVariable('q_dot', nv)
    # q_ddot = prb.createInputVariable('q_ddot', nv)
    # x_dot = utils.double_integrator(q, q_dot, q_ddot)
    # prb.setDynamics(x_dot)
    # prb.setDt(dt)

    # Create problem STATE and INPUT variables
    q = prb.createStateVariable("q", nq)
    qdot = prb.createInputVariable("qdot", nv)

    # Create dynamics
    prb.setDynamics(qdot)
    prb.setDt(dt)

    # q_init = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    q_init = cs.SX.zeros(nq,1)

    FD = kindyn.aba()
    ID = cs.Function.deserialize(kindyn.rnea())

    # transcription
    if solver_type != 'ilqr':
        Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

    # initial state and initial guess
    q_init = np.ones(nq) * 0.01
    q.setBounds(q_init, q_init, 0)
    qdot.setBounds(np.zeros(nv), np.zeros(nv), 0)
    q.setInitialGuess(q_init)

    # joint limits
    q_max = kindyn.q_max()
    q_min = kindyn.q_min()
    q.setBounds(lb=q_min, ub=q_max, nodes=range(1, n_nodes+1))

    from urdf_parser_py import urdf as urdf_parser
    tau_lims = []
    robot = urdf_parser.Robot.from_xml_string(urdf)
    for joint_name in joint_names:
        for joint in robot.joints:
            if joint.name == joint_name:
                tau_lims.append(joint.limit.effort)

    tau_lims = np.array(tau_lims)

    # # cartesian target
    # frame = 'TCP_gripper_A' #'end_effector'# frame = 'TCP_gripper_A'
    # FK = cs.Function.deserialize(kindyn.fk(frame))
    # DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

    # p = FK(q=q)['ee_pos']
    # p_start = FK(q=q_init)['ee_pos']  # .toarray()

    # p_tgt = np.array([0.0,0.0,0.0])  # p_start.copy()
    # p_tgt[0] = 1.2
    # p_tgt[1] = 0.2
    # p_tgt[2] = 0.0

    # v = DFK(q=q, qdot=qdot)['ee_vel_linear']

    # quat = FK(q=q)['ee_rot']    

    # quat_start = FK(q=q_init)['ee_rot']  # .toarray()
    # quat_tgt = np.array([0.0,0.0,0.0])  # quat_start.copy()

    # z_versor = quat[:,2]
    # plane_normal = np.array([0,0,1])
    # dot_product = cs.dot(plane_normal, z_versor)

    ##### Cost function #####
    # Cost function (min velocity)
    prb.createIntermediateCost("qdot", 1e-3*cs.sumsqr(qdot))
    # Cost function (min effort)
    id = cs.Function.deserialize(kindyn.rnea())
    gcomp = id(q, 0, 0)
    # gcomp = prb.createVariable("gcomp", nq)
    prb.createFinalCost("max_gcomp", -1*cs.sumsqr(gcomp))

    # create solver
    prb_solver = solver.Solver.make_solver(
        solver_type, 
        prb,
        opts={
            'ipopt.tol': 1e-4,
            'ipopt.max_iter': 3000
            }
        )

    # solver
    tic = time.time()
    prb_solver.solve()
    toc = time.time()
    print('time elapsed solving:', toc - tic)

    solution = prb_solver.getSolutionDict()

    q_hist = solution["q"]
    q_last = q_hist.T[-1].T
    print(q_last)

    gcomp_max = id(q_last, cs.SX.zeros(nv,1), cs.SX.zeros(nv,1))
    print(gcomp_max)

    tau = gcomp_max
    tau_rated = 81.0 # 174.0 # 81.0
    tau_available = []
    for torque in tau.nonzeros():
        if torque <= tau_rated and torque >= 0.0:
            tau_available.append(tau_rated - torque)
        elif torque >= -tau_rated and torque < 0.0:
            tau_available.append(-tau_rated - torque)
        else:
            tau_available.append(0.0)
    tau_available = cs.DM(tau_available)
    print("Torque available on each joint", tau_available)

    J = cs.Function.deserialize(kindyn.jacobian(tip_link, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    J_z = J(q=q_last)['J'][2:3,:]
    J_z_g = J_z * 9.81
    available_payloads = tau_available/ J_z_g.T

    # jac = kdl_to_mat(kin_mdl.jacobian(q))
    # j_z = np.array(jac[2:3])
    # # print("Jacobian along z axis", j_z)
    # j_z_g = j_z * 9.81
    # available_payloads = np.divide(tau_available, j_z_g)

    print("available payloads on each joint", available_payloads)
    # taking the absolute value should be redundant. only to avoid small values of J_z (numeric error) to make for huge payloads
    real_payload = np.amin(np.absolute(available_payloads)) 
    print("The Payload: ", real_payload)

    if not is_online:
        replay_trajectory(dt, kindyn.joint_names()[1:], q_hist).replay(is_floating_base=False)

    return real_payload


online = True
if not online:
    from modular.URDF_writer import UrdfWriter
    
    #create UrdfWriter object adn joint map to store homing values
    urdf_writer = UrdfWriter(speedup=True)
    homing_joint_map = {}
    # J1
    # data = urdf_writer.add_module('concert/module_joint_concert_yaw_ORANGE_B.yaml', 0, False)
    data = urdf_writer.add_module('module_joint_yaw_ORANGE_B.yaml', 0, False)
    homing_joint_map[str(data['lastModule_name'])] = {'angle': 0.0}
    # J2
    # data = urdf_writer.add_module('concert/module_joint_concert_elbow_ORANGE_B.yaml', 0, False)
    data = urdf_writer.add_module('module_joint_double_elbow_ORANGE_B.yaml', 0, False)
    homing_joint_map[str(data['lastModule_name'])] = {'angle': 0.5}
    # J4
    # data = urdf_writer.add_module('concert/module_joint_concert_elbow_ORANGE_B.yaml', 0, False)
    data = urdf_writer.add_module('module_joint_double_elbow_ORANGE_B.yaml', 0, False)
    homing_joint_map[data['lastModule_name']] = {'angle': 1.0}
    # J3
    # data = urdf_writer.add_module('concert/module_joint_concert_yaw_ORANGE_B.yaml', 0, False)
    data = urdf_writer.add_module('module_joint_yaw_ORANGE_B.yaml', 0, False)
    homing_joint_map[str(data['lastModule_name'])] = {'angle': 0.0}
    # J6
    # data = urdf_writer.add_module('concert/module_joint_concert_elbow_ORANGE_B.yaml', 0, False)
    data = urdf_writer.add_module('module_joint_double_elbow_ORANGE_B.yaml', 0, False)
    homing_joint_map[str(data['lastModule_name'])] = {'angle': 1.5}
    # # J5
    # # data = urdf_writer.add_module('concert/module_joint_concert_yaw_ORANGE_B.yaml', 0, False)
    # data = urdf_writer.add_module('module_joint_yaw_ORANGE_B.yaml', 0, False)
    # homing_joint_map[str(data['lastModule_name'])] = {'angle': 0.0}

    data = urdf_writer.add_module('module_gripper_B.yaml', 0, False)
    # urdf_writer.add_simple_ee(0.0, 0.0, 0.189, 0.0)

    urdf = urdf_writer.process_urdf()
    urdf_writer.write_urdf()

    if __name__ == '__main__':
        tip_link_hardcoded = 'TCP_gripper_A'
        compute_payload(urdf, tip_link_hardcoded, online)
   
else:
    from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
    app = Flask(__name__)


    # call URDF_writer.py to modify the urdf
    @app.route('/getPayload/', methods=['GET'])
    def changeURDF():
        urdf = request.form.get('result')
        print(urdf)
        tip_link = request.form.get('tip_link')
        print(tip_link)
        payload = compute_payload(urdf, tip_link, online)
        print(payload)
        return "{:.2f}".format(payload)

    if __name__ == '__main__':
        main()

