import casadi as cs
import numpy as np
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.functions import RecedingConstraint, RecedingCost

def _barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))

# todo this is a composition of atomic tasks: how to do?
class ContactTask:
    def __init__(self, name, prb, kin_dyn, frame, force, nodes=None):
        """
        establish/break contact
        """
        # todo name can be part of action
        self.prb = prb
        self.name = name
        self.frame = frame
        self.initial_nodes = [] if nodes is None else nodes
        # todo add in opts
        self.fmin = 10.

        # todo: force should be retrieved from frame!!!!!!!!!!
        self.force = force
        self.kin_dyn = kin_dyn

        # ======== initialize constraints ==========
        # todo are these part of the contact class? Should they belong somewhere else?
        # todo auto-generate constraints here given the method (inheriting the name of the method
        self.constraints = list()
        self._zero_vel_constr = self._zero_velocity(self.initial_nodes) # this is easily a cartesianTask
        self._unil_constr = self._unilaterality(self.initial_nodes)
        self._friction_constr = self._friction(self.initial_nodes)

        self.constraints.append(self._zero_vel_constr)
        self.constraints.append(self._unil_constr)
        self.constraints.append(self._friction_constr)
        # ===========================================

        self.nodes = []
        # initialize contact nodes
        # todo default action?
        # todo probably better to keep track of nodes, divided by action
        self.actions = []
        # setNodes(nodes1) ---> self.actions.append(nodes1)
        # setNodes(nodes2) ---> self.actions.append(nodes2)
        # self.actions = [[nodes1], [nodes2]]
        # setNodes(nodes3, reset)
        # self.actions = [[nodes3]]
        # self.removeAction()
        # should I keep track of these?
        # self.lift_nodes = []
        # self.contact_nodes = []
        # self.unilat_nodes = []
        # self.zero_force_nodes = []
        # self.contact_nodes = list(range(1, self.prb.getNNodes()))# all the nodes
        # self.unilat_nodes = list(range(self.prb.getNNodes() - 1))
        # todo reset all the other "contact" constraints on these nodes
        # self._reset_contact_constraints(self.action.frame, nodes_in_horizon_x)

    def setNodes(self, nodes):

        self.nodes = nodes
        all_nodes = list(range(self.prb.getNNodes()))
        self._reset(all_nodes)

        # if it's on:
        nodes_on_x = [k for k in self.nodes if k <= self.prb.getNNodes() - 1]
        nodes_on_u = [k for k in self.nodes if k < self.prb.getNNodes() - 1]

        nodes_off_x = [k for k in all_nodes if k not in nodes_on_x]
        nodes_off_u = [k for k in all_nodes if k not in nodes_on_u and k < self.prb.getNNodes() - 1]

        # todo F=0 and v=0 must be activated on the same node otherwise there is one interval where F!=0 and v!=0

        # setting the nodes
        erasing = True
        self._zero_vel_constr.setNodes(nodes_on_x, erasing=erasing)  # state + starting from node 1
        self._unil_constr.setNodes(nodes_on_u, erasing=erasing)
        # self._friction_constr[self.frame].setNodes(nodes_on_u, erasing=erasing)  # input

        f = self.force
        fzero = np.zeros(f.getDim())
        f.setBounds(fzero, fzero, nodes_off_u)

        print(f'contact {self.name} nodes:')
        print(f'zero_velocity: {self._zero_vel_constr.getNodes().tolist()}')
        print(f'unilaterality: {self._unil_constr.getNodes().tolist()}')
        print(f'force: ')
        print(f'{np.where(self.force.getLowerBounds()[0, :] == 0.)[0].tolist()}')
        print(f'{np.where(self.force.getUpperBounds()[0, :] == 0.)[0].tolist()}')
        print('===================================')

    def _zero_velocity(self, nodes=None):
        """
        equality constraint
        """
        active_nodes = [] if nodes is None else nodes

        # todo what if I don't want to set a reference? Does the parameter that I create by default weigthts on the problem?
        cartesian_constr = CartesianTask('zero_velocity', self.prb, self.kin_dyn, self.frame, nodes=self.initial_nodes, indices=[0, 1, 2], cartesian_type='velocity')
        constr = cartesian_constr.getConstraint()
        # dfk = cs.Function.deserialize(self.kin_dyn.frameVelocity(self.frame, self.kd_frame))
        # todo how do I find that there is a variable called 'v' which represent velocity?
        # ee_v = dfk(q=self.prb.getVariables('q'), qdot=self.prb.getVariables('v'))['ee_vel_linear']
        #
        # constr = self.prb.createConstraint(f"{self.frame}_vel", ee_v, nodes=[])
        return constr

    def _unilaterality(self, nodes=None):
        """
        barrier cost
        """
        active_nodes = [] if nodes is None else nodes

        fcost = _barrier(self.force[2] - self.fmin)

        # todo or createIntermediateCost?
        barrier = self.prb.createCost(f'{self.frame}_unil_barrier', 1e2 * fcost, nodes=active_nodes)
        return barrier

    def _friction(self, nodes=None):
        """
        barrier cost
        """
        active_nodes = [] if nodes is None else nodes

        f = self.force
        mu = 0.5
        fcost = _barrier(f[2] ** 2 * mu ** 2 - cs.sumsqr(f[:2]))
        barrier = self.prb.createIntermediateCost(f'{self.frame}_fc', 1e-3 * fcost, nodes=active_nodes)
        return barrier

    def _reset(self, nodes):

        # todo reset task
        # task.reset()
        for fun in self.constraints:
            ## constraints and variables --> relax bounds
            if isinstance(fun, RecedingConstraint):
                ## constraints and variables --> relax bounds
                c_inf = np.inf * np.ones(fun.getDim())
                fun.setBounds(-c_inf, c_inf, nodes)
            elif isinstance(fun, RecedingCost):
                current_nodes = fun.getNodes().astype(int)
                new_nodes = current_nodes.copy()
                for val in nodes:
                    new_nodes = new_nodes[new_nodes != val]
                fun.setNodes(new_nodes, erasing=True)

        self.force.setBounds(lb=np.full(self.force.getDim(), -np.inf),
                             ub=np.full(self.force.getDim(), np.inf))

    def getNodes(self):
        return self.nodes

    def getName(self):
        return self.name

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