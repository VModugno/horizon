import casadi as cs
from typing import List, Iterable, Union, Sequence
import random, string
from horizon.problem import Problem
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
from dataclasses import dataclass, field


def generate_id() -> str:
    id_len = 6
    return ''.join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(id_len))

@dataclass
class Task:
    # todo this is context: transform to context
    prb: Problem
    kin_dyn: pycasadi_kin_dyn.CasadiKinDyn
    kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

    # todo: there should be also a type
    type: str
    name: str
    fun_type: str = 'constraint'
    weight: Union[List, float] = 1.0
    nodes: Sequence = field(default_factory=list)
    indices: Union[List, np.ndarray] = np.array([0, 1, 2]).astype(int)
    id: str = field(init=False, default_factory=generate_id)

    @classmethod
    def from_dict(cls, task_dict):
        return cls(**task_dict)

    def __post_init__(self):
        # todo: this is for simplicity
        self.indices = np.array(self.indices)

    def setNodes(self, nodes):
        self.nodes = nodes
        self.n_active = len(self.nodes)

    def setIndices(self, indices):
        self.indices = indices

    def _reset(self):
        pass

    def getNodes(self):
        return self.nodes

    def getName(self):
        return self.name

    def getType(self):
        return self.type