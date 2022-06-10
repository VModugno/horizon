import casadi as cs
from horizon.problem import Problem
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn

# todo name is useless
class Task:
    def __init__(self, name, prb: Problem, kin_dyn, frame, nodes=None, dim=None, weight=None, kd_frame=None):

        # todo name can be part of action
        self.name = name
        self.prb = prb
        self.kin_dyn = kin_dyn
        self.frame = frame
        self.nodes = None

        if dim is None:
            self.dim = np.array([0, 1, 2]).astype(int)
        else:
            self.dim = np.array(dim)

        self.weight = 1. if weight is None else weight
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED if kd_frame is None else kd_frame

        self.initial_nodes = [] if nodes is None else nodes

    def setNodes(self, nodes):

        self.nodes = nodes
        self.n_active = len(self.nodes)

    def _reset(self):
        pass

    def getNodes(self):
        return self.nodes

    def getName(self):
        return self.name
