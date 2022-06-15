from horizon.rhc.tasks.task import Task
import casadi as cs
from horizon.problem import Problem
import numpy as np

# todo name is useless
class PosturalTask(Task):
    def __init__(self, prb: Problem, kin_dyn, task_node):
        super().__init__(prb, kin_dyn, task_node)

        # if 'postural_ref' not in task_node:
        #     raise ValueError('Postural reference is not set')

        self.q0_ref = task_node['postural_ref'][self.indices]
        self.fun_type = 'constraint' if 'fun_type' not in task_node else task_node['fun_type']

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createResidual

        self._initialize()

    def _initialize(self):
        self.q = self.prb.getVariables('q')
        name_fun = f'postural_{self.name}'# '_'.join(map(str, self.indices))
        self.q0 = self.prb.createParameter(f'{name_fun}_tgt', self.indices.size)
        self.fun = self.instantiator(f'{name_fun}_task', self.weight * (self.q[self.indices] - self.q0), nodes=self.nodes)
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