from horizon.rhc.tasks.task import Task
from horizon.problem import Problem
import numpy as np

class InteractionTask(Task):
    def __init__(self, frame, force, *args, **kwargs):

        self.frame = frame
        super().__init__(*args, **kwargs)

        self.indices = np.array([0, 1, 2, 3, 4, 5]).astype(int) if self.indices is None else np.array(self.indices).astype(int)

        self.f = force[self.indices]
        self._initialize()

    def _initialize(self):
        # ===========================================
        self.actions = []
        # todo: this is not the way to retrieve the force
        # self.f = self.prb.getVariables('f_' + self.frame)[self.indices]

        # fzero = np.zeros(self.f.getDim())
        # self.f.setBounds(fzero, fzero, self.nodes)

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