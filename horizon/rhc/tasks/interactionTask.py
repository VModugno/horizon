from abc import abstractmethod
from horizon.rhc.tasks.task import Task
from horizon.problem import Problem
import numpy as np
from typing import Callable, Dict, Any, List
import casadi as cs

class InteractionTask(Task):
    def __init__(self, frame, *args, **kwargs):

        self.frame = frame
        super().__init__(*args, **kwargs)

        # TODO what to do with it?
        self.indices = np.array([0, 1, 2, 3, 4, 5]).astype(int) if self.indices is None else np.array(self.indices).astype(int)

        self._initialize()

    def _initialize(self):
        # ===========================================
        self.actions = []
        # todo: this is not the way to retrieve the force
        # self.f = self.prb.getVariables('f_' + self.frame)[self.indices]

        # fzero = np.zeros(self.f.getDim())
        # self.f.setBounds(fzero, fzero, self.nodes)
    
    @abstractmethod
    def getWrench(self):
        pass

    @abstractmethod
    def setInitialGuess(self, f0, nodes=None):
        # todo tentative, re-think
        pass

    def getFrame(self):
        return self.frame

    # todo crazy misleading name
    def setNodes(self, nodes):
        super().setNodes(nodes)

        self.nodes = nodes
        self._reset()
        self._set_zero(nodes)

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _set_zero(self, nodes):
        pass


class SurfaceContact(InteractionTask):
    def __init__(self, frame, dimensions, *args, **kwargs):

        # init base
        super().__init__(frame, *args, **kwargs)

        # create input (todo: support degree > 0)
        self.wrench = self.prb.createInputVariable('f_' + frame, dim=6)

        # register wrench with model
        self.model.setContactFrame(frame, self.wrench)

    def getWrench(self):
        return self.wrench

    def setInitialGuess(self, f0, nodes=None):
        self.wrench.setInitialGuess(f0, nodes)

    def _reset(self):
        # todo reset only on given nodes
        self.wrench.setBounds(lb=np.full(self.wrench.getDim(), -np.inf),
                              ub=np.full(self.wrench.getDim(), np.inf))

    def _set_zero(self, nodes):
        self.wrench.setBounds(lb=np.full(self.wrench.getDim(), 0),
                              ub=np.full(self.wrench.getDim(), 0),
                              nodes=nodes)


class VerticesContact(InteractionTask):
    def __init__(self, frame, vertex_frames, *args, **kwargs):

        # init base
        super().__init__(frame, *args, **kwargs)

        # create input (todo: support degree > 0)
        # self.prb.createInputVariable('f_' + frame, dimensions)


class CopConstraint(Task):
    def __init__(self, 
                 subtask: Dict[str, Task], 
                 dimensions: List[float],
                 shape='box',
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.contact = Task.subtask_by_class(subtask, InteractionTask)
        if self.contact is None:
            raise ValueError(f'InteractionTask subtask required')
        if isinstance(self.contact, SurfaceContact):
            self._make_surface_cop(self.contact, dimensions, shape)
        else:
            pass
    
    def _make_surface_cop(self, contact, dimensions, shape):
        
        # get wrench from child
        f = contact.getWrench()

        # compute rotation matrix
        frame = contact.getFrame()
        _, R = self.model.fk(frame)

        # turn to local coord
        f_local = cs.vertcat(
            R.T @ f[0:3],
            R.T @ f[3:6]
        )

        # write constraint
        xmin, xmax = -dimensions[0], dimensions[0]
        ymin, ymax = -dimensions[1], dimensions[1]

        M_cop = cs.DM.zeros(4, 6)
        M_cop[:, 2] = [xmin, -xmax, ymin, -ymax]
        M_cop[[0, 1], 4] = [1, -1]
        M_cop[[2, 3], 3] = [-1, 1]

        rot_M_cop = M_cop @ f_local

        # add constraint
        cop_cnsrt = self.prb.createIntermediateConstraint(f'cop_{frame}', rot_M_cop)
        cop_cnsrt.setLowerBounds(-np.inf * np.ones(4))
            
