from horizon.rhc.tasks.task import Task
from horizon.problem import Problem
import numpy as np

class InteractionTask(Task):
    # todo this should be general, not action-dependent
    def __init__(self, prb: Problem, kin_dyn, task_node):
        super().__init__(prb, kin_dyn, task_node)

        self._initialize()

    def _initialize(self):
        # ===========================================
        self.actions = []

        self.f = self.prb.getVariables('f_' + self.frame)[self.indices]
        fzero = np.zeros(self.f.getDim())
        self.f.setBounds(fzero, fzero, self.nodes)

    def setNodes(self, nodes):
        super().setNodes(nodes)

        self.nodes = nodes
        self._reset()

        fzero = np.zeros(self.f.getDim())
        self.f.setBounds(fzero, fzero, nodes)

    def _reset(self):
        # todo reset only on given nodes
        self.f.setBounds(lb=np.full(self.f.getDim(), -np.inf),
                         ub=np.full(self.f.getDim(), np.inf))