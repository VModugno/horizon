import casadi as cs
from horizon.problem import Problem
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn

# todo name is useless
class Task:
    def __init__(self, prb: Problem, kin_dyn, task_node):

        task_fun_type = None if 'fun_type' not in task_node else task_node['fun_type']

        self.prb = prb
        self.kin_dyn = kin_dyn
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED if 'kd_frame' not in task_node else task_node['kd_frame']
        self.name = task_node['name']
        self.frame = None if 'frame' not in task_node else task_node['frame']
        self.nodes = [] if 'nodes' not in task_node else task_node['nodes']
        self.indices = np.array([0, 1, 2]).astype(int) if 'indices' not in task_node else np.array(task_node['indices'])
        self.weight = 1.0 if 'weight' not in task_node else task_node['weight']

    def setNodes(self, nodes):

        self.nodes = nodes
        self.n_active = len(self.nodes)

    def _reset(self):
        pass

    def getNodes(self):
        return self.nodes

    def getName(self):
        return self.name
