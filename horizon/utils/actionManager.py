import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs
from horizon.problem import Problem

# barrier function
def _barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))

def _trj(tau):
    return 64.*tau**3*(1-tau)**3

def compute_trajectory(start, goal, node_duration, node_duration_in_horizon):
    # compute swing trj
    trj = None

    for k in node_duration_in_horizon:
        tau = (k - k_start) / node_duration
        trj = _trj(tau) * s.clearance
        trj += (1 - tau) * start + tau * goal

    return trj


class Step:
    """
    simple class representing a step, contains the main info about the step
    """
    def __init__(self, leg: int, k_start: int, k_goal: int,
                 start=np.array([]), goal=np.array([]),
                 clearance=0.10):
        self.leg = leg
        self.k_start = k_start
        self.k_goal = k_goal
        self.goal = np.array(goal)
        self.start = np.array(start)
        self.clearance = clearance



class ActionManager:
    """
    set of actions which involves combinations of constraints and bounds
    """
    def __init__(self, prb: Problem, contacts_map, opts=None):

        self.prb = prb
        self.opts = opts
        self.N = self.prb.getNNodes() - 1
        # todo list of contact is fixed?
        self.current_cycle = 0 # what to do here?


        self.contacts = contacts_map.keys()
        self.nc = len(self.contacts)

        self.forces = contacts_map

        self.contact_nodes = [list(range(1, self.N + 1)) for _ in range(self.nc)]
        self.unilat_nodes = [list(range(self.N)) for _ in range(self.nc)]
        self.clea_nodes = [list() for _ in range(self.nc)]
        self.contact_k = [list() for _ in range(self.nc)]


    def init_constraints(self):
        self._zero_vel_constr = dict()
        self._unil_constr = dict()
        self._friction_constr = dict()
        self._foot_z_constr = dict()
        self._target_constr = dict()

        for frame in self.contacts:
            self._zero_vel_constr[frame] = self._zero_velocity(frame)
            self._unil_constr[frame] = self._unilaterality(frame)
            self._friction_constr[frame] = self._friction(frame)
            self._target_constr[frame] = self._cartesian_task(frame)
            self._foot_z_constr[frame] = self._cartesian_task(frame)


    # todo:
    #  def getFramePos():
    #  def addFrame(self, frame):  ---> what is this?

    def _reset_contact_constraints(self, frame, nodes):
        contact_constraints = [self._zero_vel_constr, self._unil_constr, self._friction_constr, self._foot_z_constr, self.forces, self._target_constr]
        for c in contact_constraints:
            if c[frame]:
                ## constraints and variables --> relax bounds
                c_inf = np.inf * np.ones(c.getDim())
                c[frame].setBounds(-c_inf, c_inf, nodes)
                ## todo should implement a function to reset to default values

    def _zero_velocity(self, frame):
        raise NotImplementedError()

    def _unilaterality(self, frame):
        raise NotImplementedError()

    def _friction(self, frame):
        raise NotImplementedError()

    def _cartesian_task(self, frame, dim=3):
        raise NotImplementedError()

    def setContact(self, on, frame, nodes, target=None):
        """
        add an end-effector to the contact list
        """
        # todo reset all the other "contact" constraints on these nodes
        self._reset_contact_constraints(frame, nodes)

        # todo contact_nodes, unil_consr, friction_constr are slightly different
        if on == 1:
            erasing = True
            self._zero_vel_constr[frame].setNodes(nodes, erasing=erasing) # state + starting from node 1
            self._unil_constr[frame].setNodes(nodes, erasing=erasing) # input -> without last node
            self._friction_constr[frame].setNodes(nodes, erasing=erasing) # input

            if target is not None:
                self._target_constr[frame].setNodes(nodes, erasing=erasing)

        elif on == 0:
            f = self.forces[frame]
            fzero = np.zeros(f.getDim())
            f.setBounds(fzero, fzero)

    def setStep(self, step: Step):
        """
        add step to horizon stack
        """

        # todo how to define current cycle
        k0 = 0
        N = self.N
        l = step.leg
        k_start = step.k_start - k0
        k_goal = step.k_goal - k0
        swing_nodes = list(range(k_start, k_goal))
        swing_nodes_in_horizon_x = [k for k in swing_nodes if k >= 0 and k <= N]
        swing_nodes_in_horizon_u = [k for k in swing_nodes if k >= 0 and k < N]
        n_swing = len(swing_nodes)

        # this step is outside the horizon!
        if n_swing == 0:
            return 0

        # update nodes contact constraint
        self.contact_nodes[l] = [k for k in self.contact_nodes[l] if k not in swing_nodes and k < self.N]

        # update nodes for unilateral constraint
        self.unilat_nodes[l] = [k for k in self.unilat_nodes[l] if k not in swing_nodes]


        if len(swing_nodes_in_horizon_x) > 0:
            self.clea_nodes[l].extend(swing_nodes_in_horizon_x)

        print('contact_nodes', self.contact_nodes)
        print('unilat_nodes', self.unilat_nodes)
        print('clea_nodes', self.clea_nodes)

        # self.setContact(frame, nodes)
        # self._foot_z_constr[frame].setNodes(nodes, erasing=True)

        # compute z_trajectory
        # trj = compute_trajectory(step.start, step.goal, n_swing, swing_nodes_in_horizon_x)
        # self.foot_tgt_params[frame].assign([0, 0, trj], nodes=k)

        # zero force
        # f = self.forces[frame]
        # fzero = np.zeros(f.getDim())
        # f.setBounds(fzero, fzero)


class ActionManagerImpl(ActionManager):
    def __init__(self, prb: Problem, opts):
        super().__init__(prb, opts)

        forces = None # todo implement
        self.joint_pos = self.prb.getVariables('q')
        self.joint_vel = self.prb.getVariables('q_dot')

        fixed_joint_map = None
        self.kd = pycasadi_kin_dyn.CasadiKinDyn(self.urdf, fixed_joints=fixed_joint_map)
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

        f0 = opts.get('f0', np.array([0, 0, 250]))
        self.fmin = opts.get('fmin', 0)
        self.steps = list()

        self.forces = dict(frames=forces)

    def _zero_force(self, frame):
        """
        equality constraint
        """
        f = self.forces[frame]
        constr = self.prb.createConstraint(f'{f.getName()}_force', f, nodes=[])
        return constr

    def _zero_velocity(self, frame):
        """
        equality constraint
        """
        dfk = cs.Function.deserialize(self.kd.frameVelocity(frame, self.kd_frame))
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

    def _cartesian_task(self, frame, dim=3):
        """
        equality constraint
        """
        fk = cs.Function.deserialize(self.kd.framePosition(frame, self.kd_frame))
        ee_p = fk(q=self.joint_pos)['ee_vel_linear']

        # todo or in problem or here check name of variables and constraints
        task = self.prb.createParameter(f'{frame}_tgt', 3)
        constr = self.prb.createConstraint(f"{frame}_task", ee_p[dim] - task[dim], nodes=[])
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
    prb = Problem(ns)
    contacts = ['l_foot', 'r_foot']
    am = ActionManager(prb, contacts)

    k_start = 4
    k_end = 8
    s = Step(0, k_start, k_end)

    print(am.contact_nodes)
    print(am.unilat_nodes)
    print(am.clea_nodes)
    print(am.contact_k)


    am.setStep(s)