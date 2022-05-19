import copy

import numpy as np
import casadi as cs
from horizon.problem import Problem
import casadi_kin_dyn.py3casadi_kin_dyn as pycasadi_kin_dyn
from horizon.functions import RecedingConstraint, RecedingCost
from horizon import misc_function as misc

def _barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))


class Task:
    def __init__(self, prb):
        pass

    def cartesianTask(self, frame, nodes):
        raise NotImplementedError()

    def contact(self, frame, nodes):
        raise NotImplementedError()


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
        self.nodes = None

    def setRef(self, ref_traj):
        self.ref = np.atleast_2d(ref_traj)

    def setNodes(self, nodes):

        # todo manage better
        if not nodes:
            self.reset()
            return 0

        self.nodes = nodes
        self.n_active = len(self.nodes)

        # todo when to activate manual mode?
        if self.ref.shape[1] != self.n_active:
            raise ValueError(f'Wrong goal dimension inserted: ({self.ref.shape[1]} != {self.n_active})')

        self.pos_constr.setNodes(self.nodes, erasing=True)  # <==== SET NODES
        self.pos_tgt.assign(self.ref, nodes=self.nodes) # <==== SET TARGET

        print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        print('===================================')

    def getNodes(self):
        return self.nodes

    def reset(self):

        self.nodes = []
        self.pos_constr.setNodes(self.nodes, erasing=True)


class Contact:
    # todo this should be general, not action-dependent
    # activate() # disable() # recede() -----> setNodes()
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

        self.nodes = []
        # initialize contact nodes
        # todo default action?
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
        return self.nodes()
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