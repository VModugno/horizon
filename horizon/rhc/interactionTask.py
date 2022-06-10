import casadi as cs
from horizon.problem import Problem
from horizon.functions import RecedingConstraint, RecedingCost
import numpy as np

class InteractionTask:
    # todo this should be general, not action-dependent
    def __init__(self, name, prb: Problem, kin_dyn, frame, nodes=None, dim=None, task_fun_type=None, weight=None):
        """
        establish/break contact
        """

        # todo what about f is already there? add a constraint or add bounds?

        # todo name can be part of action
        self.prb = prb
        self.name = name
        self.frame = frame
        self.weight = 1.0 if weight is None else weight
        self.initial_nodes = [] if nodes is None else nodes
        # todo not used right now
        self.kin_dyn = kin_dyn

        # ===========================================
        self.nodes = []
        self.actions = []

        self.f = self.prb.getVariables('f_' + frame)[dim]
        fzero = np.zeros(self.f.getDim())
        self.f.setBounds(fzero, fzero, self.initial_nodes)

    def setNodes(self, nodes):

        self.nodes = nodes
        all_nodes = list(range(self.prb.getNNodes()))
        self._reset(all_nodes)

        fzero = np.zeros(self.f.getDim())
        self.f.setBounds(fzero, fzero, nodes)

    def _reset(self, nodes):
        # todo reset only on given nodes
        self.f.setBounds(lb=np.full(self.f.getDim(), -np.inf),
                         ub=np.full(self.f.getDim(), np.inf))

    def getNodes(self):
        return self.nodes

    def getName(self):
        return self.name