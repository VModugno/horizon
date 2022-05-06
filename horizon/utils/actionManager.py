import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs
from horizon.problem import Problem
import os
from horizon.functions import Constraint, Cost
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

    def __init__(self, prb: Problem, urdf, kindyn, contacts_map, opts=None):

        self.prb = prb
        self.opts = opts
        self.N = self.prb.getNNodes() - 1
        # todo list of contact is fixed?
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

        # set default z-pos of contact
        self.default_foot_z = dict()
        for frame in contacts:
            self.default_foot_z[frame] = 0.

        self.k0 = 0

        self.init_constraints()
        self._init_default_action()

    def recede(self):
        self.k0 = self.k0 + 1
        print(self.k0)

    def set_polynomial_trajectory(self, start, goal, nodes, param):

        nodes_in_horizon = [k for k in nodes if k >= 0 and k <= self.N]

        for k in nodes_in_horizon:
            tau = (k - k_start) / len(nodes)
            trj = _trj(tau) * s.clearance
            trj += (1 - tau) * start + tau * goal
            param.assign([0, 0, trj], nodes=k)

    def _init_default_action(self):

        # todo for now the default is robot still, in contact
        # contact nodes
        self.contact_nodes = {contact: list(range(1, self.N + 1)) for contact in contacts}  # all the nodes
        self.unilat_nodes = {contact: list(range(self.N)) for contact in contacts}
        self.clea_nodes = {contact: list() for contact in contacts}
        self.contact_k = {contact: list() for contact in contacts}

        print('init:')
        print('contact_nodes', self.contact_nodes)
        print('unilat_nodes', self.unilat_nodes)
        print('clea_nodes', self.clea_nodes)
        print('contact_k', self.contact_k)
        print('===================================')

    def init_constraints(self):

        # todo ISSUE: I need to initialize all the constraints that I will use in the problem
        self._zero_vel_constr = dict()
        self._unil_constr = dict()
        self._friction_constr = dict()
        self._cartesian_constr = dict()
        self._target_constr = dict()
        self._foot_z_constr = dict()
        self._foot_tgt_params = dict()
        self._foot_z_param = dict()

        for frame in self.contacts:
            self._zero_vel_constr[frame] = self._zero_velocity(frame)
            self._unil_constr[frame] = self._unilaterality(frame)
            self._friction_constr[frame] = self._friction(frame)
            self._foot_tgt_params[frame], self._foot_tgt_params[frame] = self._cartesian_task('foot_tgt', frame)
            self._foot_z_constr[frame], self._foot_z_param[frame] = self._cartesian_task('foot_z', frame)
    #
    # todo:
    #  def getFramePos():
    #  def addFrame(self, frame):  ---> what is this?

    def _reset_contact_constraints(self, frame, nodes):
        contact_constraints = [self._zero_vel_constr, self._unil_constr, self._friction_constr, self._foot_z_constr,
                               self.forces, self._target_constr]
        for c in contact_constraints:
            if frame in c:
                fun = c[frame]
                if isinstance(fun, Constraint):
                    ## constraints and variables --> relax bounds
                    c_inf = np.inf * np.ones(fun.getDim())
                    fun.setBounds(-c_inf, c_inf, nodes)
                elif isinstance(fun, Cost):
                    current_nodes = fun.getNodes().astype(int)
                    new_nodes = np.delete(current_nodes, nodes)
                    fun.setNodes(new_nodes)

                ## todo should implement --> removeNodes()
                ## todo should implement a function to reset to default values

    def _zero_velocity(self, frame):
        raise NotImplementedError()

    def _unilaterality(self, frame):
        raise NotImplementedError()

    def _friction(self, frame):
        raise NotImplementedError()

    def _cartesian_task(self, name, frame, dim=3):
        raise NotImplementedError()

    def setContact(self, on, frame, nodes, target=None):
        """
        establish/break contact
        """
        nodes_now = np.array(nodes) - self.k0

        # todo prepare nodes of contact on/off:
        nodes_in_horizon_x = [k for k in nodes_now if k >= 0 and k <= self.N]
        nodes_in_horizon_u = [k for k in nodes_now if k >= 0 and k < self.N]

        # todo reset all the other "contact" constraints on these nodes
        self._reset_contact_constraints(frame, nodes_in_horizon_x)

        # todo contact_nodes, unil_consr, friction_constr are slightly different

        if on == 1:

            # if it's on:
            # update nodes contact constraint
            # todo if nodes was a Set it would have been easier
            # todo why < self.N?
            nodes_to_add = [k for k in nodes_now if k not in self.contact_nodes[frame] and k <= self.N]
            if nodes_to_add:
                self.contact_nodes[frame].extend(nodes_to_add)

            # update nodes for unilateral constraint
            nodes_to_add = [k for k in nodes_now if k not in self.contact_nodes[frame] and k <= self.N]
            if nodes_to_add:
                self.unilat_nodes[frame].extend(nodes_to_add)

            # update clearance nodes
            self.clea_nodes[frame] = [k for k in self.clea_nodes[frame] if k not in nodes]

            erasing = True
            self._zero_vel_constr[frame].setNodes(self.contact_nodes[frame], erasing=erasing)  # state + starting from node 1
            self._unil_constr[frame].setNodes(self.unilat_nodes[frame], erasing=erasing)  # input -> without last node
            self._friction_constr[frame].setNodes(self.unilat_nodes[frame], erasing=erasing)  # input

            if target is not None:
                self._target_constr[frame].setNodes(nodes, erasing=erasing)

        elif on == 0:

            # if it's off:
            # update contact nodes
            # todo i don't understand --> k < self.N
            self.contact_nodes[frame] = [k for k in self.contact_nodes[frame] if k not in nodes and k < self.N]

            # update nodes for unilateral constraint
            self.unilat_nodes[frame] = [k for k in self.unilat_nodes[frame] if k not in nodes]

            f = self.forces[frame]
            fzero = np.zeros(f.getDim())
            f.setBounds(fzero, fzero, nodes_in_horizon_u)

    def setCartesianTask(self, frame, nodes, task, task_nodes, param, start, goal):
        """
        set cartesian task
        """
        nodes_now = np.array(nodes) - self.k0

        # todo prepare nodes of action:
        nodes_in_horizon_x = [k for k in nodes_now if k >= 0 and k <= self.N]
        nodes_in_horizon_u = [k for k in nodes_now if k >= 0 and k < self.N]

        # todo to reset?
        # self._reset_contact_constraints(frame, nodes_in_horizon_x)

        nodes_to_add = [k for k in nodes_now if k not in task_nodes and k <= self.N]
        if nodes_to_add:
            task_nodes.extend(nodes_to_add)

        # todo should be somewhere else
        task.setNodes(task_nodes, erasing=True)  # <==== SET NODES
        self.set_polynomial_trajectory(start, goal, task_nodes, param)  # <==== SET TARGET


    def setStep(self, step: Step):
        """
        add step to horizon stack
        """

        # todo how to define current cycle
        frame = step.frame
        k_start = step.k_start
        k_goal = step.k_goal
        swing_nodes = list(range(k_start, k_goal))
        n_swing = len(swing_nodes)

        # this step is outside the horizon!
        if n_swing == 0:
            return 0

        # break contact at swing nodes + z_trajectory + (optional) xy goal
        # contact
        self.setContact(0, frame, swing_nodes)
        # cartesian task:
        # z goal
        # update clearance nodes
        z_start = self.default_foot_z[frame] if s.start.size == 0 else s.start[2]
        z_goal = self.default_foot_z[frame] if s.goal.size == 0 else s.goal[2]
        self.setCartesianTask(frame, swing_nodes, self._foot_z_constr[frame], self.clea_nodes[frame], self._foot_z_param[frame], z_start, z_goal)


        # xy goal
        if k_goal <= self.N and k_goal > 0 and s.goal.size > 0:
            # todo cartesian task: what if it is a trajectory or a single point?
            self.setCartesianTask(frame, k_goal,  self.contact_k[frame], k_goal, self._foot_tgt_params[frame], s.goal)
            self.contact_k[frame].append(k_goal)  # <==== SET NODES
            self._foot_tgt_params[frame].assign(s.goal, nodes=k_goal)  # <==== SET TARGET


class ActionManagerImpl(ActionManager):
    def __init__(self, prb: Problem, urdf, kindyn, contacts_map, opts = None):
        super().__init__(prb, urdf, kindyn, contacts_map, opts)

    def _zero_force(self, frame):
        """
        equality constraint
        """
        f = self.forces[frame]
        constr = self.prb.createConstraint(f'{frame}_force', f, nodes=[])
        return constr

    def _zero_velocity(self, frame):
        """
        equality constraint
        """
        dfk = cs.Function.deserialize(self.kd.frameVelocity(frame, self.kd_frame))
        # print('daniele')
        ee_v = dfk(q=self.joint_pos, qdot=self.joint_vel)['ee_vel_linear']

        constr = self.prb.createConstraint(f"{frame}_vel", ee_v, nodes=[])
        return constr

    def _unilaterality(self, frame):
        """
        barrier cost
        """
        f = self.forces[frame]
        fcost = _barrier(f[2] - self.fmin)

        # todo or createIntermediateCost?
        barrier = self.prb.createCost(f'{frame}_unil_barrier', 1e-3 * fcost, nodes=[])
        return barrier

    def _friction(self, frame):
        """
        barrier cost
        """
        f = self.forces[frame]
        mu = 0.5
        fcost = _barrier(f[2] ** 2 * mu ** 2 - cs.sumsqr(f[:2]))
        barrier = self.prb.createIntermediateCost(f'{frame}_fc', 1e-3 * fcost, nodes=[])
        return barrier

    # todo if I want to get rid of the name, I need automatic name creation: if no name is given, create a dummy one
    def _cartesian_task(self, name, frame, dim=None):
        """
        equality constraint
        """
        if dim is None:
            dim = np.array([0, 1, 2]).astype(int)
        else:
            dim = np.array(dim)

        fk = cs.Function.deserialize(self.kd.fk(frame))
        ee_p = fk(q=self.joint_pos)['ee_pos']

        # todo or in problem or here check name of variables and constraints
        task = self.prb.createParameter(f'{name}_{frame}_tgt', dim.size)
        constr = self.prb.createConstraint(f"{name}_{frame}_task", ee_p[dim] - task, nodes=[])
        return constr, task

    # def _friction(self, frame):
    #     """
    #     inequality constraint
    #     """
    #     mu = 0.5
    #     frame_rot = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
    #     fc, fc_lb, fc_ub = self.kd.linearized_friction_cone(f, mu, frame_rot)
    #     self.prb.createIntermediateConstraint(f"f{frame}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))

    # def _unilaterality(self, f):
    #     """
    #     inequality constraint
    #     """
    #     # todo or createIntermediateConstraint?
    #     f = self.forces[frame]
    #     constr = self.prb.createConstraint(f'{f.getName()}_unil', f_z[2] - self.fmin, nodes=[])
    #     constr.setUpperBounds(np.inf)
    #     return constr


if __name__ == '__main__':
    ns = 10
    prb = Problem(ns, receding=True)
    path_to_examples = os.path.dirname('../examples/')

    urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
    urdf = open(urdffile, 'r').read()
    contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

    fixed_joint_map = None
    kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)

    nq = kd.nq()
    nv = kd.nv()
    nc = len(contacts)
    nf = 3

    q = prb.createStateVariable('q', nq)
    v = prb.createStateVariable('v', nv)
    a = prb.createInputVariable('a', nv)
    forces = [prb.createInputVariable('f_' + c, nf) for c in contacts]
    am = ActionManagerImpl(prb, urdf, kd, dict(zip(contacts, forces)))

    k_start = 4
    k_end = 8
    s = Step('lf_foot', k_start, k_end)

    am.setStep(s)
    # am.setContact(0, 'lf_foot', [8, 9, 10, 11, 12])
    # am.setContact(0, 'rf_foot', [0, 1, 2])
    # am.recede()

    print('contact_nodes', am.contact_nodes)
    print('unilat_nodes', am.unilat_nodes)
    print('clea_nodes', am.clea_nodes)
    print('contact_k', am.contact_k)
    print('===================================')

