import copy

import matplotlib.pyplot
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs
from horizon.problem import Problem
from horizon.utils import utils, kin_dyn, plotter
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers.solver import Solver
from horizon.rhc import receding_tasks
from horizon.ros import replay_trajectory
from horizon import misc_function as misc
import rospy
import os
from horizon.functions import Constraint, Cost, RecedingCost, RecedingConstraint
import subprocess

# barrier function
def _barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))


def _trj(tau):
    return 64. * tau ** 3 * (1 - tau) ** 3


class Action:
    """
    simple class representing a generic action
    """

    def __init__(self, frame: str, k_start: int, k_goal: int, start=np.array([]), goal=np.array([])):
        self.frame = frame
        self.k_start = k_start
        self.k_goal = k_goal
        self.goal = np.array(goal)
        self.start = np.array(start)


class Step(Action):
    """
    simple class representing a step, contains the main info about the step
    """

    def __init__(self, frame: str, k_start: int, k_goal: int, start=np.array([]), goal=np.array([]), clearance=0.10):
        super().__init__(frame, k_start, k_goal, start, goal)
        self.clearance = clearance


# what if the action manager provides only the nodes? but for what?
# - for the contacts
# - for the variables and the constraints
class ActionManager:
    """
    set of actions which involves combinations of constraints and bounds
    """

    def __init__(self, prb: Problem, urdf, kindyn, contacts_map, default_foot_z, opts=None):

        self.prb = prb
        self.opts = opts
        self.N = self.prb.getNNodes() - 1
        # todo list of contact is fixed?

        self.constraints = list()
        self.nodes = None
        self.current_cycle = 0  # what to do here?

        self.contacts = contacts_map.keys()
        self.nc = len(self.contacts)

        self.forces = contacts_map

        self.kd = kindyn
        # todo useless here
        self.urdf = urdf
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

        self.joint_pos = self.prb.getVariables('q')
        self.joint_vel = self.prb.getVariables('v')

        # f0 = opts.get('f0', np.array([0, 0, 250]))
        f0 = np.array([0, 0, 55])
        # self.fmin = opts.get('fmin', 0)
        self.fmin = 10.


        self.default_foot_z = default_foot_z

        self.k0 = 0

        self.init_constraints()
        self._init_default_action()

    def compute_polynomial_trajectory(self, k_start, nodes, p_start, p_goal, clearance, dim=None):

        if dim is None:
            dim = [0, 1, 2]

        # todo check dimension of parameter before assigning it
        nodes_in_horizon = [k for k in nodes if k >= 0]

        traj_array = np.zeros(len(nodes_in_horizon))

        start = p_start[dim]
        goal = p_goal[dim]

        index = 0
        for k in nodes_in_horizon:
            tau = (k - k_start) / len(nodes)
            trj = _trj(tau) * clearance
            trj += (1 - tau) * start + tau * goal
            traj_array[index] = trj
            index = index+1

        return np.array(traj_array)

    def _init_default_action(self):

        # todo for now the default is robot still, in contact
        print('===================== initializing ========================')
        print('===========================================================')
        # contact nodes
        # contact and unilat nodes are slightly different. How to define?
        for frame, c_constr in self.contact_constr.items():
            c_constr.active(list(range(self.N + 1)), 1)

        # self.contact_nodes = {contact: list(range(1, self.N + 1)) for contact in contacts}  # all the nodes
        # self.unilat_nodes = {contact: list(range(self.N)) for contact in contacts}
        # self.clea_nodes = {contact: list() for contact in contacts}
        # self.contact_k = {contact: list() for contact in contacts}

        print('===========================================================')
        print('===========================================================')

        # default action
        # for frame, cnsrt_item in self._zero_vel_constr.items():
        #     cnsrt_item.setNodes(self.contact_nodes[frame])
        #
        # for frame, cnsrt_item in self._unil_constr.items():
        #     cnsrt_item.setNodes(self.unilat_nodes[frame])


    def init_constraints(self):

        # todo ISSUE: I need to initialize all the constraints that I will use in the problem
        # self._zero_vel_constr = dict()
        # self._unil_constr = dict()
        # self._friction_constr = dict()
        # self._cartesian_constr = dict()
        # self._target_constr = dict()
        # self._foot_z_constr = dict()
        # self._foot_tgt_params = dict()
        # self._foot_z_param = dict()

        self.contact_constr = dict()
        self.z_constr = dict()
        self.foot_tgt_constr = dict()

        for frame in self.contacts:
            self.contact_constr[frame] = receding_tasks.Contact(f'{frame}_contact', self.kd, self.kd_frame, self.prb, self.forces[frame], frame)
            self.z_constr[frame] = receding_tasks.CartesianTask(f'{frame}_z_constr', self.kd, self.prb, frame, 2)
            self.foot_tgt_constr[frame] = receding_tasks.CartesianTask(f'{frame}_foot_tgt_constr', self.kd, self.prb, frame, [0, 1])



    def setContact(self, on, frame, nodes):
        """
        establish/break contact
        """
        # todo reset all the other "contact" constraints on these nodes
        # self._reset_task_constraints(frame, nodes_in_horizon_x)

        # todo what to do with z_constr?
        # self.z_constr.reset()

        self.contact_constr[frame].active(nodes, on)


    def setStep(self, step: Step):
        """
        add step to horizon stack
        """
        s = copy.deepcopy(step)
        # todo how to define current cycle
        frame = s.frame
        k_start = s.k_start
        k_goal = s.k_goal
        swing_nodes = list(range(k_start, k_goal))
        n_swing = len(swing_nodes)

        # this step is outside the horizon!
        if n_swing == 0:
            return 0

        # break contact at swing nodes + z_trajectory + (optional) xy goal
        # contact
        self.setContact(0, frame, swing_nodes)

        # xy goal
        if self.N >= k_goal > 0 and step.goal.size > 0:
            self.foot_tgt_constr[frame].activate([k_goal], s.goal[:2])

        # z goal
        start = np.array([0, 0, self.default_foot_z[frame]]) if s.start.size == 0 else s.start
        goal = np.array([0, 0, self.default_foot_z[frame]]) if s.goal.size == 0 else s.goal

        # todo this way I can define my own trajectory goal (assign the parameter)
        z_traj = self.compute_polynomial_trajectory(k_start, swing_nodes, start, goal, s.clearance, dim=2)
        self.z_constr[frame].activate(swing_nodes, z_traj)

class ActionManagerImpl(ActionManager):
    def __init__(self, prb: Problem, urdf, kindyn, contacts_map, default_foot_z, opts=None):
        super().__init__(prb, urdf, kindyn, contacts_map, default_foot_z, opts)

    def execute(self, solver):
        """
        given the default, spin once shifting the horizon
        """
        shift_num = -1
        self._update_initial_state(solver, shift_num)

        # todo check if constraint has inputs and remove last node
        # todo the default is all the contacts are active
        # default: "renewing" nodes
        # state


        # set constraints
        for c_name, c_constr in self.contact_constr.items():
            c_constr.recede(shift_num)

        for c_name, c_constr in self.z_constr.items():
            c_constr.recede(shift_num)

        for c_name, c_constr in self.foot_tgt_constr.items():
            c_constr.recede(shift_num)

        # todo
        #  the difference here is which list of nodes get new entry nodes and which do not
        #  can also shift only decaying nodes and fill everything else with default nodes
        # difference between "action", which is decaying and "default", which is renewing

    def _update_initial_state(self, solver: Solver, shift_num):

        x_opt = solver.getSolutionState()
        u_opt = solver.getSolutionInput()

        xig = np.roll(x_opt, shift_num, axis=1)

        for i in range(abs(shift_num)):
            xig[:, -1 - i] = x_opt[:, -1]
        self.prb.getState().setInitialGuess(xig)

        uig = np.roll(u_opt, shift_num, axis=1)

        for i in range(abs(shift_num)):
            uig[:, -1 - i] = u_opt[:, -1]
        self.prb.getInput().setInitialGuess(uig)

        self.prb.setInitialState(x0=xig[:, 0])


if __name__ == '__main__':

    ns = 40
    tf = 10.0
    dt = tf / ns

    prb = Problem(ns, receding=True)
    path_to_examples = os.path.dirname('../examples/')

    urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
    urdf = open(urdffile, 'r').read()
    contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

    fixed_joint_map = None
    kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
    kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

    q_init = np.array([0.0, 0.0, 0.505, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.9, -1.52,
                       0.0, 0.9, -1.52,
                       0.0, 0.9, -1.52,
                       0.0, 0.9, -1.52])

    q0 = q_init
    nq = kd.nq()
    nv = kd.nv()
    nc = len(contacts)
    nf = 3

    v0 = np.zeros(nv)
    prb.setDt(dt)

    # state and control vars
    q = prb.createStateVariable('q', nq)
    v = prb.createStateVariable('v', nv)
    a = prb.createInputVariable('a', nv)
    forces = [prb.createInputVariable('f_' + c, nf) for c in contacts]
    fmap = {k: v for k, v in zip(contacts, forces)}

    f0 = np.array([0, 0, 55])
    # dynamics ODE
    _, xdot = utils.double_integrator_with_floating_base(q, v, a)
    prb.setDynamics(xdot)

    # underactuation constraint
    id_fn = kin_dyn.InverseDynamics(kd, contacts, kd_frame)
    tau = id_fn.call(q, v, a, fmap)
    prb.createIntermediateConstraint('dynamics', tau[:6])

    # final goal (a.k.a. integral velocity control)
    ptgt_final = [0.2, 0, 0]
    vmax = [0.05, 0.05, 0.05]
    ptgt = prb.createParameter('ptgt', 3)

    # goalx = prb.createFinalResidual("final_x",  1e3*(q[0] - ptgt[0]))
    goalx = prb.createFinalConstraint("final_x", q[0] - ptgt[0])
    goaly = prb.createFinalResidual("final_y", 1e3 * (q[1] - ptgt[1]))
    goalrz = prb.createFinalResidual("final_rz", 1e3 * (q[5] - ptgt[2]))
    # base_goal_tasks = [goalx, goaly, goalrz]

    # final velocity
    # v.setBounds(v0, v0, nodes=ns)
    # regularization costs

    # base rotation
    prb.createResidual("min_rot", 1e-3 * (q[3:5] - q0[3:5]))

    # joint posture
    prb.createResidual("min_q", 1e0 * (q[7:] - q0[7:]))

    # joint velocity
    prb.createResidual("min_v", 1e-2 * v)

    # final posture
    prb.createFinalResidual("min_qf", 1e0 * (q[7:] - q0[7:]))

    # regularize input
    prb.createIntermediateResidual("min_q_ddot", 1e-1 * a)

    for f in forces:
        prb.createIntermediateResidual(f"min_{f.getName()}", 1e-2 * (f - f0))

    # costs and constraints implementing a gait schedule
    com_fn = cs.Function.deserialize(kd.centerOfMass())

    # save default foot height
    default_foot_z = dict()

    # contact velocity is zero, and normal force is positive
    for i, frame in enumerate(contacts):
        # fk functions and evaluated vars
        fk = cs.Function.deserialize(kd.fk(frame))
        dfk = cs.Function.deserialize(kd.frameVelocity(frame, kd_frame))

        ee_p = fk(q=q)['ee_pos']
        ee_rot = fk(q=q)['ee_rot']
        ee_v = dfk(q=q, qdot=v)['ee_vel_linear']

        # save foot height
        default_foot_z[frame] = (fk(q=q0)['ee_pos'][2])

        # vertical contact frame
        rot_err = cs.sumsqr(ee_rot[2, :2])
        # prb.createIntermediateCost(f'{frame}_rot', 1e-1 * rot_err)

        # todo action constraints
        # kinematic contact
        # unilateral forces
        # friction
        # clearance
        # xy goal

    am = ActionManagerImpl(prb, urdf, kd, dict(zip(contacts, forces)), default_foot_z)

    k_start = 10
    k_end = 15
    s_1 = Step('lf_foot', k_start, k_end)

    k_start = 40
    k_end = 45
    s_2 = Step('lf_foot', k_start, k_end)

    k_start = 20
    k_end = 30
    f_k = cs.Function.deserialize(kd.fk('lh_foot'))
    initial_lh_foot = f_k(q=q_init)['ee_pos']
    step_len = np.array([0.2, 0.1, 0])
    print(f'step target: {initial_lh_foot + step_len}')
    s_3 = Step('lh_foot', k_start, k_end, goal=initial_lh_foot + step_len)



    k_start = 10
    k_end = 20
    s_4 = Step('rh_foot', k_start, k_end)

    # k_start = 8
    # k_end = 15
    # s_2 = Step('rf_foot', k_start, k_end)

    Transcriptor.make_method('multiple_shooting', prb)


    # set initial condition and initial guess
    q.setBounds(q0, q0, nodes=0)
    v.setBounds(v0, v0, nodes=0)

    q.setInitialGuess(q0)

    for f in forces:
        f.setInitialGuess(f0)
    #

    # ============== add steps!!!!!!!!! =======================
    am.setStep(s_1)
    am.setStep(s_2)
    # am.setStep(s_3)

    step_pattern = ['lf_foot', 'rh_foot', 'rf_foot', 'lh_foot']
    k_step_n = 5
    k_start = 10

    step_list = list()
    n_step = 15

    for n in range(n_step):
        l = step_pattern[n % len(step_pattern)]
        k_end = k_start + k_step_n
        s = Step(l, k_start, k_end)
        print(l, k_start, k_end)
        k_start = k_end
        step_list.append(s)

    # for s_i in step_list:
    #     am.setStep(s_i)

    # am.setStep(step_list[0])
    # am.setStep(step_list[1])
    # am.setStep(step_list[2])
    # am.setStep(step_list[3])

    # create solver and solve initial seed
    # print('===========executing ...========================')

    opts = {'ipopt.tol': 0.001,
            'ipopt.constr_viol_tol': 1e-3,
            'ipopt.max_iter': 1000,
            }

    solver_bs = Solver.make_solver('ipopt', prb, opts)
    # solver_rti = Solver.make_solver('ipopt', prb, opts)

    ptgt.assign(ptgt_final, nodes=ns)

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
    repl = replay_trajectory.replay_trajectory(dt, kd.joint_names()[2:], np.array([]), {k: None for k in contacts}, kd_frame, kd)
    iteration = 0
    while True:


        # if iteration % 20 == 0:
        #     am.setStep(s_1)
            # am.setContact(0, 'rh_foot', range(5, 15))


        am.execute(solver_bs)

        solver_bs.solve()
        solution = solver_bs.getSolutionDict()


        repl.frame_force_mapping = {contacts[i]: solution[forces[i].getName()][:, 0:1] for i in range(nc)}
        repl.publish_joints(solution['q'][:, 0])
        repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
        iteration = iteration+1
        print(iteration)
    #
    # set ROS stuff and launchfile
    plot = True

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        for contact in contacts:
            FK = cs.Function.deserialize(kd.fk(contact))
            pos = FK(q=solution['q'])['ee_pos']

            plt.title(f'feet position - plane_xy')
            plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
            plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
            plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')

        plt.figure()
        for contact in contacts:
            FK = cs.Function.deserialize(kd.fk(contact))
            pos = FK(q=solution['q'])['ee_pos']

            plt.title(f'feet position - plane_xz')
            plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
            plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
            plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')

        hplt = plotter.PlotterHorizon(prb, solution)
        hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
        hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
        matplotlib.pyplot.show()




