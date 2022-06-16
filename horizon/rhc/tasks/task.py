import casadi as cs
from typing import List, Iterable, Union
import random, string
from horizon.problem import Problem
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
from dataclasses import dataclass, field

def generate_id() -> str:
    id_len = 6
    return ''.join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(id_len))

@dataclass() # frozen=True
class TaskDescription:
    type: str
    name: str
    frame: str
    fun_type: str = None
    weight: float = 1.0
    nodes: Iterable = field(default_factory=list)
    indices: Union[List, np.ndarray] = np.array([0, 1, 2]).astype(int)
    id: str = field(init=False, default_factory=generate_id)
    kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

    def setIndices(self, indices):
        self.indices = indices

    # def ...

class Task:
    def __init__(self, prb: Problem, kin_dyn, task_node: TaskDescription):

        self.prb = prb
        self.kin_dyn = kin_dyn

        # self.kd_frame = task_node.kd_frame
        # self.name = task_node.name
        # self.frame = task_node.frame
        # self.nodes = task_node.nodes
        # self.indices = task_node.indices
        # self.weight = task_node.weight

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






