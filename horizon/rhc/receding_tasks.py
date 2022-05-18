import copy

import numpy as np
import casadi as cs
from horizon.problem import Problem
import casadi_kin_dyn.py3casadi_kin_dyn as pycasadi_kin_dyn
from horizon.functions import RecedingConstraint, RecedingCost
from horizon import misc_function as misc

def _barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))

# todo if action is "elapsed" remove it from .recede() otherwise it will stay forever
# todo name is useless
class CartesianTask:
    def __init__(self, name, kin_dyn, prb: Problem, frame, dim=None):

        # todo name can be part of action
        self.prb = prb
        self.name = name
        self.frame = frame

        if dim is None:
            dim = np.array([0, 1, 2]).astype(int)
        else:
            dim = np.array(dim)

        fk = cs.Function.deserialize(kin_dyn.fk(frame))

        # kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        # dfk = cs.Function.deserialize(kin_dyn.frameVelocity(frame, kd_frame))
        # ddfk = cs.Function.deserialize(kin_dyn.frameAcceleration(frame, kd_frame))

        # todo this is bad
        q = self.prb.getVariables('q')
        # v = self.prb.getVariables('v')

        ee_p = fk(q=q)['ee_pos']
        # ee_v = dfk(q=q, qdot=v)['ee_vel_linear']
        # ee_a = ddfk(q=q, qdot=v)['ee_acc_linear']

        # todo or in problem or here check name of variables and constraints
        pos_frame_name = f'{self.name}_{self.frame}_pos'
        # vel_frame_name = f'{self.name}_{self.frame}_vel'
        # acc_frame_name = f'{self.name}_{self.frame}_acc'

        self.pos_tgt = prb.createParameter(f'{pos_frame_name}_tgt', dim.size)
        # self.vel_tgt = prb.createParameter(f'{vel_frame_name}_tgt', dim.size)
        # self.acc_tgt = prb.createParameter(f'{acc_frame_name}_tgt', dim.size)

        self.pos_constr = prb.createConstraint(f'{pos_frame_name}_task', ee_p[dim] - self.pos_tgt, nodes=[])
        # self.vel_constr = prb.createConstraint(f'{vel_frame_name}_task', ee_v[dim] - self.vel_tgt, nodes=[])
        # self.acc_constr = prb.createConstraint(f'{acc_frame_name}_task', ee_a[dim] - self.acc_tgt, nodes=[])

        self.ref = None
        self.active_ref = None
        self.action_nodes = list()
        self.active_nodes = None
        self.n_action = None

    def activate(self, nodes, ref_traj):

        # todo centralize checking of nodes if in horizon, now is scattered everywhere
        # adding all the nodes to action nodes, even if they are not in horizon
        for k in nodes:
            if k not in self.action_nodes:
                self.action_nodes.append(k)

        # adding to active_nodes only nodes in horizon
        # self.active_nodes = [k for k in nodes if k >= 0 and k < self.prb.getNNodes()]
        self.active_nodes = [k for k in self.action_nodes if k >= 0 and k <= self.prb.getNNodes() - 1]

        self.n_action = len(self.active_nodes)

        # todo probably not very efficient
        if self.ref is None:
            self.ref = np.atleast_2d(ref_traj)
        else:
            self.ref = np.append(self.ref, np.atleast_2d(ref_traj), 1)

        self.active_ref = self.ref[:, -self.n_action:]

        # todo when to activate manual mode?
        if self.active_ref.shape[1] != self.n_action:
            raise ValueError(f'Wrong goal dimension inserted: ({self.active_ref.shape[1]} != {self.n_action})')

        self.pos_constr.setNodes(self.active_nodes, erasing=True)  # <==== SET NODES
        self.pos_tgt.assign(self.active_ref, nodes=self.active_nodes) # <==== SET TARGET

        print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        print('===================================')

    def recede(self, ks):

        # if nodes is None, task is not active
        if self.action_nodes is None:
            self.active_nodes = None
            self.n_action = None
            self.ref = None
            return 0

        # todo recede action nodes and set new active nodes
        self.action_nodes = [x + ks for x in self.action_nodes]
        self.active_nodes = [k for k in self.action_nodes if k >= 0 and k < self.prb.getNNodes()]
        # todo this is basically repeated code from activate
        self.pos_constr.setNodes(self.active_nodes, erasing=True)  # <==== SET NODES

        # if nodes is empty, reset and return
        if not np.array(self.active_nodes).size:
            self.n_action = None
            # todo right now the non-active nodes of the parameter gets dirty,
            #  because .assing() only assign a value to the current nodes, the other are left with the old value
            #  better to reset?
            # self.pos_tgt.reset()
            return 0

        # get only the right portion of param
        self.active_ref = self.ref[:, -len(self.active_nodes):]
        self.pos_tgt.assign(self.active_ref, nodes=self.active_nodes)

        print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues().tolist()}')
        print(f'param task {self.name} nodes:', self.active_ref.tolist())

    def setTarget(self, ref_traj):
        """
        manually overrides the goal
        """
        self.ref = ref_traj

    def reset(self):

        self.active_nodes = []
        self.pos_constr.setNodes(self.active_nodes, erasing=True)



class Contact():

    # todo this should be general, not action-dependent
    # then one can activate/disable
    # activate() # disable() # recede() #
    def __init__(self, name, kin_dyn, kd_frame, prb, force, frame):
        """
        establish/break contact
        """
        # todo name can be part of action
        self.prb = prb
        self.name = name
        self.force = force
        self.frame = frame
        # todo add in opts
        self.fmin = 10.

        self.kin_dyn = kin_dyn
        self.kd_frame = kd_frame

        # ======== initialize constraints ==========
        # todo are these part of the contact class? Should they belong somewhere else?
        self.constraints = list()
        self._zero_vel_constr = self._zero_velocity()
        self._unil_constr = self._unilaterality()
        self._friction_constr = self._friction()

        self.constraints.append(self._zero_vel_constr)
        self.constraints.append(self._unil_constr)
        self.constraints.append(self._friction_constr)
        # ===========================================

        # initialize contact nodes
        # todo default action?
        # should I keep track of these?
        self.lift_nodes = []
        self.contact_nodes = []
        self.unilat_nodes = []
        # self.contact_nodes = list(range(1, self.prb.getNNodes()))# all the nodes
        # self.unilat_nodes = list(range(self.prb.getNNodes() - 1))
        # todo reset all the other "contact" constraints on these nodes
        # self._reset_contact_constraints(self.action.frame, nodes_in_horizon_x)

    def active(self, nodes, on):

        # todo prepare nodes of contact on/off:
        nodes_in_horizon_x = [k for k in nodes if k >= 0 and k <= self.prb.getNNodes() - 1]
        nodes_in_horizon_u = [k for k in nodes if k >= 0 and k < self.prb.getNNodes() - 1]

        # reset contact / unilaterality / friction
        self._reset_constraints_and_force(nodes_in_horizon_x)

        if on == 1:
            # if it's on:
            # update nodes contact constraint
            # simply add to the contact_nodes the new nodes
            nodes_to_add = [k for k in nodes if k not in self.contact_nodes and k <= self.prb.getNNodes() - 1]
            if nodes_to_add:
                self.contact_nodes.extend(nodes_to_add)

            nodes_to_add = [k for k in nodes if k not in self.unilat_nodes and k < self.prb.getNNodes() - 1]
            # update nodes for unilateral constraint
            # simply add to the unilat_nodes the new nodes
            if nodes_to_add:
                self.unilat_nodes.extend(nodes_to_add)

        elif on == 0:

            self.lift_nodes.extend(nodes)
            erasing = True
            # if it's off:
            # update contact nodes
            # todo F=0 and v=0 must be activated on the same node otherwise there is one interval where F!=0 and v!=0
            self.contact_nodes = [k for k in self.contact_nodes if k not in nodes and k <= self.prb.getNNodes() - 1]
            # update nodes for unilateral constraint
            self.unilat_nodes = [k for k in self.unilat_nodes if k not in nodes]

            # set forces to zero
            f = self.force
            fzero = np.zeros(f.getDim())
            f.setBounds(fzero, fzero, nodes_in_horizon_u)

        erasing = True
        self._zero_vel_constr.setNodes(self.contact_nodes, erasing=erasing)  # state + starting from node 1
        self._unil_constr.setNodes(self.unilat_nodes, erasing=erasing)
        # self._friction_constr[self.frame].setNodes(self.unilat_nodes[self.frame], erasing=erasing)  # input

        print(f'contact {self.name} nodes:')
        print(f'zero_velocity: {self._zero_vel_constr.getNodes().tolist()}')
        print(f'unilaterality: {self._unil_constr.getNodes().tolist()}')
        # print(f'force: imma here but im difficult to show')
        print(f'force: ')
        print(f'{np.where(self.force.getLowerBounds()[0, :] == 0.)[0].tolist()}')
        print(f'{np.where(self.force.getUpperBounds()[0, :] == 0.)[0].tolist()}')
        print('===================================')

    def _zero_velocity(self):
        """
        equality constraint
        """
        dfk = cs.Function.deserialize(self.kin_dyn.frameVelocity(self.frame, self.kd_frame))
        # todo how do I find that there is a variable called 'v' which represent velocity?
        ee_v = dfk(q=self.prb.getVariables('q'), qdot=self.prb.getVariables('v'))['ee_vel_linear']

        constr = self.prb.createConstraint(f"{self.frame}_vel", ee_v, nodes=[])
        return constr

    def _unilaterality(self):
        """
        barrier cost
        """
        fcost = _barrier(self.force[2] - self.fmin)

        # todo or createIntermediateCost?
        barrier = self.prb.createCost(f'{self.frame}_unil_barrier', 1e-3 * fcost, nodes=[])
        return barrier

    def _friction(self):
        """
        barrier cost
        """
        f = self.force
        mu = 0.5
        fcost = _barrier(f[2] ** 2 * mu ** 2 - cs.sumsqr(f[:2]))
        barrier = self.prb.createIntermediateCost(f'{self.frame}_fc', 1e-3 * fcost, nodes=[])
        return barrier

    def _reset_constraints_and_force(self, nodes):

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

    def recede(self, ks):

        print("=============================================== ======== ===============================================")
        print("=============================================== RECEDING ===============================================")
        print("=============================================== ======== ===============================================")
        # update nodes for contact constraints
        new_node_x = [] if self.prb.getNNodes() - 1 in self.lift_nodes else [self.prb.getNNodes() - 1]
        new_node_u = [] if self.prb.getNNodes() - 2 in self.lift_nodes else [self.prb.getNNodes() - 2]
        self.lift_nodes = [x + ks for x in self.lift_nodes]

        # todo check if it is to add the new node or not
        shifted_contact_nodes = [x + ks for x in self.contact_nodes] + new_node_x
        self.contact_nodes = [x for x in shifted_contact_nodes if x > 0]

        # update nodes for unilateral constraint
        shifted_unilat_nodes = [x + ks for x in self.unilat_nodes] + new_node_u
        self.unilat_nodes = [x for x in shifted_unilat_nodes if x > 0]

        erasing = True
        self._zero_vel_constr.setNodes(self.contact_nodes, erasing=erasing)  # state + starting from node 1
        self._unil_constr.setNodes(self.unilat_nodes, erasing=erasing)

        # update nodes (bounds) for force
        shifted_lb = misc.shift_array(self.force.getLowerBounds(), ks, -np.inf)
        shifted_ub = misc.shift_array(self.force.getUpperBounds(), ks, np.inf)

        self.force.setLowerBounds(shifted_lb)
        self.force.setUpperBounds(shifted_ub)


        print(f'contact {self.name} nodes:')
        print(f'zero_velocity: {self._zero_vel_constr.getNodes().tolist()}')
        print(f'unilaterality: {self._unil_constr.getNodes().tolist()}')
        print(f'force: ')
        print(f'{np.where(self.force.getLowerBounds()[0, :] == 0.)[0].tolist()}')
        print(f'{np.where(self.force.getUpperBounds()[0, :] == 0.)[0].tolist()}')
        print('===================================')

        # alternative method
        # for constr in self.constraints:
        #     nodes = constr.getNodes()
        #     shifted_nodes = [x + kd for x in nodes] + [self.prb.getNNodes()]
        #     self.contact_nodes = [x for x in shifted_nodes if x > 0]



            # elif isinstance(fun, Cost):
            #     current_nodes = fun.getNodes().astype(int)
            #     new_nodes = np.delete(current_nodes, nodes)
            #     fun.setNodes(new_nodes)

                ## todo should implement --> removeNodes()
                ## todo should implement a function to reset to default values

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