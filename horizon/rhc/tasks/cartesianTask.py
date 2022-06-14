from horizon.rhc.tasks.task import Task
import casadi as cs
from horizon.problem import Problem
import numpy as np

# todo name is useless
class CartesianTask(Task):
    def __init__(self, name, prb: Problem, kin_dyn, frame, nodes=None, indices=None, weight=None, kd_frame=None, fun_type=None, cartesian_type=None):
        super().__init__(name, prb, kin_dyn, frame, nodes, indices, weight, kd_frame)

        self.instantiator = self.prb.createConstraint if fun_type is None else fun_type
        self.cartesian_type = 'position' if cartesian_type is None else cartesian_type

        if fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif fun_type == 'cost':
            self.instantiator = self.prb.createResidual

        self._initialize()

    def _initialize(self):
        q = self.prb.getVariables('q')
        v = self.prb.getVariables('v')

        # kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        if self.cartesian_type == 'position':
            fk = cs.Function.deserialize(self.kin_dyn.fk(self.frame))
            ee_p_t = fk(q=q)['ee_pos']
            ee_p_r = fk(q=q)['ee_rot']
            # ee_p = cs.vertcat(ee_p_t, ee_p_r)

            frame_name = f'{self.name}_{self.frame}_pos'
            self.pos_tgt = self.prb.createParameter(f'{frame_name}_tgt', self.indices.size)
            fun = ee_p_t[self.indices] - self.pos_tgt
        elif self.cartesian_type == 'velocity':
            dfk = cs.Function.deserialize(self.kin_dyn.frameVelocity(self.frame, self.kd_frame))
            ee_v_t = dfk(q=q, qdot=v)['ee_vel_linear']
            ee_v_r = dfk(q=q, qdot=v)['ee_vel_angular']
            ee_v = cs.vertcat(ee_v_t, ee_v_r)

            frame_name = f'{self.name}_{self.frame}_vel'
            self.vel_tgt = self.prb.createParameter(f'{frame_name}_tgt', self.indices.size)
            fun = ee_v[self.indices] - self.vel_tgt
        elif self.cartesian_type == 'acceleration':
            ddfk = cs.Function.deserialize(self.kin_dyn.frameAcceleration(self.frame, self.kd_frame))
            ee_a_t = ddfk(q=q, qdot=v)['ee_acc_linear']
            ee_a_r = ddfk(q=q, qdot=v)['ee_acc_angular']
            ee_a = cs.vertcat(ee_a_t, ee_a_r)
            frame_name = f'{self.name}_{self.frame}_acc'
            self.acc_tgt = self.prb.createParameter(f'{frame_name}_tgt', self.indices.size)
            fun = ee_a[self.indices] - self.acc_tgt

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
        super().setNodes(nodes)

        if not nodes:
            self.nodes = []
            self.constr.setNodes(self.nodes, erasing=True)
            return 0

        # todo when to activate manual mode?
        if self.ref.shape[1] != self.n_active:
            raise ValueError(f'Wrong nodes dimension inserted: ({self.ref.shape[1]} != {self.n_active})')

        if self.cartesian_type == 'position':
            tgt = self.pos_tgt
        elif self.cartesian_type == 'velocity':
            tgt = self.vel_tgt
        elif self.cartesian_type == 'acceleration':
            tgt = self.acc_tgt
        else:
            raise Exception('Wrong cartesian type')

        # core
        tgt.assign(self.ref, nodes=self.nodes) # <==== SET TARGET
        self.constr.setNodes(self.nodes, erasing=True)  # <==== SET NODES

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')