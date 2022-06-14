from horizon.rhc.tasks.task import Task
import casadi as cs
from horizon.problem import Problem
import numpy as np

# todo name is useless
class PosturalTask(Task):
    def __init__(self, name, prb: Problem, kin_dyn, frame, nodes=None, indices=None, weight=None, kd_frame=None, fun_type=None, postural_ref=None):
        super().__init__(name, prb, kin_dyn, frame, nodes, indices, weight, kd_frame)

        if postural_ref is None:
            raise ValueError('Postural reference is not set')

        self.q0_ref = postural_ref
        self.instantiator = self.prb.createConstraint if fun_type is None else fun_type

        if fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif fun_type == 'cost':
            self.instantiator = self.prb.createResidual

        self._initialize()

    def _initialize(self):
        self.q = self.prb.getVariables('q')
        name_fun = f'postural_{self.name}'# '_'.join(map(str, self.indices))
        self.q0 = self.prb.createParameter(f'{name_fun}_tgt', self.indices.size)
        self.fun = self.instantiator(f'{name_fun}_task', self.weight * (self.q[self.indices] - self.q0), nodes=self.initial_nodes)
        self.q0.assign(self.q0_ref)

    def getConstraint(self):
        return self.fun

    def setRef(self, postural_ref):

        ref_matrix = np.atleast_2d(postural_ref)

        if ref_matrix.shape[0] != self.fun.getDim():
            raise ValueError(f'Wrong goal dimension inserted: ({ref_matrix.shape[0]} != {self.fun.getDim()})')

        self.q0_ref = ref_matrix

        self.q0.assign(self.q0_ref)

    def setNodes(self, nodes):
        super().setNodes(nodes)

        if not nodes:
            self.nodes = []

        self.fun.setNodes(self.nodes, erasing=True)  # <==== SET NODES

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')