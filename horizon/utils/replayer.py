import time

import numpy as np

from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions import integrators
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

def plotFunction(name, dim=None):
    cnstr_val = solution[name]['val'][0, 0]
    cnstr_lb = solution[name]['lb'][0, 0]
    cnstr_ub = solution[name]['ub'][0, 0]

    var_dim_select = set(range(cnstr_val.shape[0]))

    if dim is not None:
        if not set(dim).issubset(var_dim_select):
            raise Exception('Wrong selected dimension.')
        else:
            var_dim_select = dim

    plt.figure()
    for dim in var_dim_select:
        plt.plot(range(cnstr_val.shape[1]), cnstr_val[dim, :])

        plt.plot(range(cnstr_lb.shape[1]), cnstr_lb[dim, :], marker="x", markersize=3, linestyle='dotted', linewidth=1)
        plt.plot(range(cnstr_ub.shape[1]), cnstr_ub[dim, :], marker="x", markersize=3, linestyle='dotted', linewidth=1)

    plt.title(f'{name}')

def checkBounds(name, tol=0.):
    checkLowerBounds(name, tol)
    checkUpperBounds(name, tol)

def checkLowerBounds(name, tol=0.):
    cnstr_val = solution[name]['val'][0, 0]
    cnstr_lb = solution[name]['lb'][0, 0]
    cnstr_nodes = solution[name]['nodes'][0, 0]


    check_lb = cnstr_lb - cnstr_val

    for dim in range(check_lb.shape[0]):
        for node in range(check_lb.shape[1]):
            if check_lb[dim, node] > tol:
                print(f"{name} violates the lower bounds at dim {dim} and node {cnstr_nodes[0, node]} of {check_lb[dim, node]}" )


def checkUpperBounds(name, tol=0.):
    cnstr_val = solution[name]['val'][0, 0]
    cnstr_ub = solution[name]['ub'][0, 0]
    cnstr_nodes = solution[name]['nodes'][0, 0]

    check_ub = cnstr_val - cnstr_ub

    for dim in range(check_ub.shape[0]):
        for node in range(check_ub.shape[1]):
            if check_ub[dim, node] > tol:
                print(f"{name} violates the upper bounds at dim {dim} and node {cnstr_nodes[0, node]} of {check_ub[dim, node]}")



transcription_method = 'multiple_shooting'  # direct_collocation
transcription_opts = dict(integrator='RK4')

urdffile = '/home/francesco/catkin_ws/external/casadi_horizon/horizon/examples/urdf/spot.urdf'
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

tot_time = 1
dt_hint = 0.02
duration_step = 0.5

n_nodes = int(tot_time / dt_hint)
n_nodes_step = int(duration_step / dt_hint)

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

jump_height = 0.1
node_start_step = 15
node_end_step = node_start_step + n_nodes_step

ms = mat_storer.matStorer('../examples/spot_wheelie.mat')
solution = ms.load()
print([name for name in solution])
exit()
constraint_names = ['multiple_shooting', 'inverse_dynamics', 'final_velocity', 'lf_foot_vel_before_lift', 'lf_foot_vel_after_lift', 'lf_foot_fc_before_lift', 'lf_foot_fc_after_lift', 'lf_foot_no_force_during_lift', 'lift_lf_foot_leg', 'land_lf_foot_leg', 'rf_foot_vel_before_lift', 'rf_foot_vel_after_lift', 'rf_foot_fc_before_lift', 'rf_foot_fc_after_lift', 'rf_foot_no_force_during_lift', 'lift_rf_foot_leg', 'land_rf_foot_leg', 'lh_foot_vel_before_lift', 'lh_foot_vel_after_lift', 'lh_foot_fc_before_lift', 'lh_foot_fc_after_lift', 'lh_foot_no_force_during_lift', 'lift_lh_foot_leg', 'land_lh_foot_leg', 'rh_foot_vel_before_lift', 'rh_foot_vel_after_lift', 'rh_foot_fc_before_lift', 'rh_foot_fc_after_lift', 'rh_foot_no_force_during_lift', 'lift_rh_foot_leg', 'land_rh_foot_leg']

contacts_name = {'lf_foot', 'rf_foot', 'lh_foot', 'rh_foot'}
contact_map = dict(zip(contacts_name, [solution['f0'], solution['f1'], solution['f2'], solution['f3']]))

# checking bounds
# for cnsrt in constraint_names:
#     checkBounds(cnsrt, tol=0.0001)

# print(solution['rh_foot_no_force_during_lift']['val'][0][0][2, :])
resample = False

if resample:
    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

    if 'dt' in solution:
        dt_before_res = solution['dt'].flatten()
    else:
        dt_before_res = solution['dt']

    dt_res = 0.001

    q = cs.SX.sym('q', n_q)
    q_dot = cs.SX.sym('q_dot', n_v)
    q_ddot = cs.SX.sym('q_ddot', n_v)
    x, x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)

    dae = {'x': x, 'p': q_ddot, 'ode': x_dot, 'quad': 1}
    q_res, qdot_res, qddot_res, contact_map_res, tau_res = resampler_trajectory.resample_torques(
        solution["q"], solution["q_dot"], solution["q_ddot"], dt_before_res, dt_res, dae, contact_map,
        kindyn,
        cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

    repl = replay_trajectory(dt_res, joint_names, q_res, contact_map_res, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
    repl.sleep(1.)
    repl.replay(is_floating_base=True)
# plotting

plt.figure()
for dim in range(solution['q'].shape[0]):
    plt.plot(range(solution['q'].shape[1]), np.array(solution['q'][dim, :]))
plt.title('q')

plt.figure()
for dim in range(solution['q_dot'].shape[0]):
    plt.plot(range(solution['q_dot'].shape[1]), np.array(solution['q_dot'][dim, :]))
plt.title('q_dot')

plt.figure()
for dim in range(solution['q_ddot'].shape[0]):
    plt.plot(range(solution['q_ddot'].shape[1]), np.array(solution['q_ddot'][dim, :]))
plt.title('q_ddot')


pos_contact_list = list()
for contact in contacts_name:
    FK = cs.Function.deserialize(kindyn.fk(contact))
    pos = FK(q=solution['q'])['ee_pos']
    plt.figure()
    plt.title(contact)
    for dim in range(n_f):
        plt.plot(np.array([range(pos.shape[1])]), np.array(pos[dim, :]), marker="x", markersize=3, linestyle='dotted')

    plt.vlines([node_start_step, node_end_step], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], linestyles='dashed', colors='k', linewidth=0.4)

#plane_xy
plt.figure()
for contact in contacts_name:
    FK = cs.Function.deserialize(kindyn.fk(contact))
    pos = FK(q=solution['q'])['ee_pos']

    plt.title(f'plane_xy')
    plt.scatter(np.array(pos[0, :]), np.array(pos[1, :]), linewidth=0.1)

# plane_xz
plt.figure()
for contact in contacts_name:
    FK = cs.Function.deserialize(kindyn.fk(contact))
    pos = FK(q=solution['q'])['ee_pos']

    plt.title(f'plane_xz')
    plt.scatter(np.array(pos[0, :]), np.array(pos[2, :]), linewidth=0.1)

# forces
for f in [f'f{i}' for i in range(len(contacts_name))]:
    plt.figure()
    for dim in range(solution[f].shape[0]):
        plt.plot(np.array(range(solution[f].shape[1])), solution[f][dim, :])

    plt.title(f'force {f}')

# dt
if 'dt' in solution:
    plt.figure()
    for dim in range(solution['dt'].shape[0]):
        plt.plot(np.array(range(solution['dt'].shape[1])), solution['dt'][dim, :])

        plt.title(f'dt')

    dt_timeline = list()
    dt_timeline.append(0)
    for i in range(solution['dt'].shape[1]):
        dt_timeline.append(dt_timeline[-1] + solution['dt'][0, i])

    for f in [f'f{i}' for i in range(len(contacts_name))]:

        plt.figure()
        for dim in range(solution[f].shape[0]):
            plt.plot(dt_timeline[:-1], solution[f][dim, :])
            plt.scatter(dt_timeline[:-1], solution[f][dim, :])
        plt.title(f'forces {f} with dt')

    # for cnstr in constraint_names:
    #     plotFunction(cnstr)


plt.show()

# resampling = True
# if resampling:
#     prb = problem.Problem(n_nodes)
#     q = prb.createStateVariable('q', n_q)
#     q_dot = prb.createStateVariable('q_dot', n_v)
#     q_ddot = prb.createInputVariable('q_ddot', n_v)
#     x, x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
#
#
#     dt_res = 0.001
#     dae = {'x': x, 'p': q_ddot, 'ode': x_dot, 'quad': 1}
#     q_res, qdot_res, qddot_res, contact_map_res, tau_res = resampler_trajectory.resample_torques(
#         solution["q"], solution["q_dot"], solution["q_ddot"], solution['dt'].flatten(), dt_res, dae, contact_map, kindyn,
#         cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)




# ======================================================
