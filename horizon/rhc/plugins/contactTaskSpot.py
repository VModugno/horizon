import casadi as cs
import numpy as np
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.functions import RecedingConstraint, RecedingCost
from horizon.utils.utils import barrier as barrier_fun
from horizon.rhc.tasks.task import Task

# todo this is a composition of atomic tasks: how to do?

class ContactTaskSpot(Task):
    def __init__(self, subtask, *args, **kwargs):
        """
        establish/break contact
        """

        # todo what if I want a default value for these subtasks?
        # if 'Force' not in subtasks:

        self.interaction_task = subtask['Force']
        self.cartesian_task = subtask['Cartesian']
        super().__init__(*args, **kwargs)

        self.force = self.interaction_task.f
        self.frame = self.cartesian_task.distal_link

        # todo: this is not the right way, as I'm not sure that f_ + self.frame is the right force
        # self.force = self.prb.getVariables('f_' + self.frame)
        # todo add in opts
        self.fmin = 10.
        # ======== initialize constraints ==========
        # todo are these part of the contact class? Should they belong somewhere else?
        # todo auto-generate constraints here given the method (inheriting the name of the method
        self.constraints = list()
        self._unil_constr = self._unilaterality(self.nodes)
        self._friction_constr = self._friction(self.nodes)

        self.constraints.append(self._unil_constr)
        self.constraints.append(self._friction_constr)
        # ===========================================

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
        self._reset()

        all_nodes = list(range(self.prb.getNNodes()))

        # if it's on:
        nodes_on_x = [k for k in self.nodes if k <= self.prb.getNNodes() - 1]
        nodes_on_u = [k for k in self.nodes if k < self.prb.getNNodes() - 1]

        # if it's off:
        nodes_off_x = [k for k in all_nodes if k not in nodes_on_x]
        nodes_off_u = [k for k in all_nodes if k not in nodes_on_u and k < self.prb.getNNodes() - 1]

        # todo F=0 and v=0 must be activated on the same node otherwise there is one interval where F!=0 and v!=0

        # setting the nodes
        erasing = True
        self.interaction_task.setNodes(nodes_off_u)  # this is from taskInterface
        self.cartesian_task.setNodes(nodes_on_x)  # state + starting from node 1  # this is from taskInterface
        self._unil_constr.setNodes(nodes_on_u, erasing=erasing) # this is from horizon
        # self._friction_constr[self.frame].setNodes(nodes_on_u, erasing=erasing)  # input

        # todo I could use InteractionTask here
        # f = self.force
        # fzero = np.zeros(f.getDim())
        # f.setBounds(fzero, fzero, nodes_off_u)

        # print(f'contact {self.name} nodes:')
        # print(f'zero_velocity: {self._zero_vel_constr.getNodes().tolist()}')
        # print(f'unilaterality: {self._unil_constr.getNodes().tolist()}')
        # print(f'force: ')
        # print(f'{np.where(self.force.getLowerBounds()[0, :] == 0.)[0].tolist()}')
        # print(f'{np.where(self.force.getUpperBounds()[0, :] == 0.)[0].tolist()}')
        # print('===================================')

    # def _force(self, nodes=None):
    #
    #     # todo this is a problem, because I have to create it like this from here. Cannot use task_interface right?
    #     task_node = {'name': f'interaction_{self.frame}', 'frame': self.frame, 'force': self.force, 'nodes': self.nodes, 'indices': [0, 1, 2]}
    #     context = {'prb': self.prb, 'kin_dyn': self.kin_dyn}
    #     interaction_constr = InteractionTask(**context, **task_node)
    #     return interaction_constr

    # def _zero_velocity(self, nodes=None):
    #     """
    #     equality constraint
    #     """
    #     active_nodes = [] if nodes is None else nodes
    #
    #     # todo what if I don't want to set a reference? Does the parameter that I create by default weigthts on the problem?
    #     task_node = {'name': 'zero_velocity', 'frame': self.frame, 'nodes': self.nodes, 'indices': [0, 1, 2], 'cartesian_type': 'velocity'}
    #     context = {'prb': self.prb, 'kin_dyn': self.kin_dyn}
    #     cartesian_constr = CartesianTask(**context, **task_node)
    #     constr = cartesian_constr.getConstraint()
    #     # dfk = cs.Function.deserialize(self.kin_dyn.frameVelocity(self.frame, self.kd_frame))
    #     # todo how do I find that there is a variable called 'v' which represent velocity?
    #     # ee_v = dfk(q=self.prb.getVariables('q'), qdot=self.prb.getVariables('v'))['ee_vel_linear']
    #     #
    #     # constr = self.prb.createConstraint(f"{self.frame}_vel", ee_v, nodes=[])
    #     # todo this returns a constraint. Do I want this?
    #     return constr

    def _unilaterality(self, nodes=None):
        """
        barrier cost
        """
        active_nodes = [] if nodes is None else nodes

        fcost = barrier_fun(self.force[2] - self.fmin)

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
        fcost = barrier_fun(f[2] ** 2 * mu ** 2 - cs.sumsqr(f[:2]))
        barrier = self.prb.createIntermediateCost(f'{self.frame}_fc', 1e-3 * fcost, nodes=active_nodes)
        return barrier

    def _reset(self):
        nodes = list(range(self.prb.getNNodes()))
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


# required for the plugin to be registered
def register_task_plugin(factory) -> None:
    factory.register("Contact", ContactTaskSpot)

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