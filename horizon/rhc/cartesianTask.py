import casadi as cs
from horizon.problem import Problem
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn

# todo name is useless
class CartesianTask:
    def __init__(self, name, prb: Problem, kin_dyn, frame, nodes=None, dim=None, cartesian_type=None, fun_type=None, weight=None, kd_frame=None):

        # todo name can be part of action
        self.prb = prb
        self.name = name
        self.frame = frame
        self.initial_nodes = [] if nodes is None else nodes
        self.instantiator = prb.createConstraint if fun_type is None else fun_type
        self.cartesian_type = 'position' if cartesian_type is None else cartesian_type
        self.weight = 1. if weight is None else weight
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED if kd_frame is None else kd_frame
        self.ref = None
        self.nodes = None

        if fun_type == 'constraint':
            self.instantiator = prb.createConstraint
        elif fun_type == 'cost':
            self.instantiator = prb.createResidual


        if dim is None:
            dim = np.array([0, 1, 2]).astype(int)
        else:
            dim = np.array(dim)

        q = self.prb.getVariables('q')

        # kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        if self.cartesian_type == 'position':
            fk = cs.Function.deserialize(kin_dyn.fk(frame))
            ee_p = fk(q=q)['ee_pos']
            frame_name = f'{self.name}_{self.frame}_pos'
            self.pos_tgt = prb.createParameter(f'{frame_name}_tgt', dim.size)
            fun = ee_p[dim] - self.pos_tgt
        elif self.cartesian_type == 'velocity':
            v = self.prb.getVariables('v')
            dfk = cs.Function.deserialize(kin_dyn.frameVelocity(frame, self.kd_frame))
            ee_v = dfk(q=q, qdot=v)['ee_vel_linear']
            frame_name = f'{self.name}_{self.frame}_vel'
            self.vel_tgt = prb.createParameter(f'{frame_name}_tgt', dim.size)
            fun = ee_v[dim] - self.vel_tgt
        elif self.cartesian_type == 'acceleration':
            v = self.prb.getVariables('v')
            ddfk = cs.Function.deserialize(kin_dyn.frameAcceleration(frame, self.kd_frame))
            ee_a = ddfk(q=q, qdot=v)['ee_acc_linear']
            frame_name = f'{self.name}_{self.frame}_acc'
            self.acc_tgt = prb.createParameter(f'{frame_name}_tgt', dim.size)
            fun = ee_a[dim] - self.acc_tgt

        self.constr = self.instantiator(f'{frame_name}_task', self.weight * fun, nodes=self.initial_nodes)
        # todo should I keep track of the nodes here?
        #  in other words: should be setNodes resetting?

    def getConstraint(self):
        return self.constr


    def setRef(self, ref_traj):

        # if self.cartesian_type == 'position':
        #     cnsrt = self.pos_constr
        # elif self.cartesian_type == 'velocity':
        #     cnsrt = self.vel_constr
        # elif self.cartesian_type == 'acceleration':
        #     cnsrt = self.acc_constr

        ref_matrix = np.atleast_2d(ref_traj)

        # if ref_matrix.shape[0] != cnsrt.getDim():
        #     raise ValueError(f'Wrong goal dimension inserted: ({ref_matrix.shape[0]} != {cnsrt.getDim()})')

        self.ref = ref_matrix

    def setNodes(self, nodes):

        # todo manage better
        # todo probably better to merge setNodes and setRef!
        if not nodes:
            self.reset()
            return 0

        self.nodes = nodes
        self.n_active = len(self.nodes)

        # todo when to activate manual mode?
        if self.ref.shape[1] != self.n_active:
            raise ValueError(f'Wrong nodes dimension inserted: ({self.ref.shape[1]} != {self.n_active})')

        if self.cartesian_type == 'position':
            self.pos_constr.setNodes(self.nodes, erasing=True)  # <==== SET NODES
            self.pos_tgt.assign(self.ref, nodes=self.nodes) # <==== SET TARGET
        elif self.cartesian_type == 'velocity':
            self.vel_constr.setNodes(self.nodes, erasing=True)
            self.vel_tgt.assign(self.ref, nodes=self.nodes)
        elif self.cartesian_type == 'acceleration':
            self.acc_constr.setNodes(self.nodes, erasing=True)
            self.acc_tgt.assign(self.ref, nodes=self.nodes)

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')

    def getNodes(self):
        return self.nodes

    def reset(self):

        self.nodes = []
        self.pos_constr.setNodes(self.nodes, erasing=True)

    def getName(self):
        return self.name
