from horizon.problem import Problem
from horizon.utils import utils, kin_dyn
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.rhc.tasks.posturalTask import PosturalTask
from horizon.rhc.tasks.contactTaskMirror import ContactTask
from horizon.rhc.tasks.limitsTask import JointLimitsTask
from typing import List, Dict
import numpy as np


class ModelDescription:
    def __init__(self, problem, model):
        self.prb = problem
        self.kd = model

    def generateModel(self, model_type=None):

        self.model_type = 'whole_body' if model_type is None else model_type

        # todo choose
        if self.model_type == 'whole_body':

            self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

            self.nq = self.kd.nq()
            self.nv = self.kd.nv()

            # custom choices
            self.nf = 3
            self.q = self.prb.createStateVariable('q', self.nq)
            self.v = self.prb.createStateVariable('v', self.nv)
            self.a = self.prb.createInputVariable('a', self.nv)

        else:
            raise NotImplementedError()

        # self.forces = [self.prb.createInputVariable('f_' + c, self.nf) for c in contacts]
        self.contacts = []
        self.fmap = dict()

        # dynamics

    def setContactFrame(self, contact):
        self.contacts.append(contact)
        f_c = self.prb.createInputVariable('f_' + contact, self.nf)
        self.fmap[contact] = f_c

    def setDynamics(self):
        _, self.xdot = utils.double_integrator_with_floating_base(self.q, self.v, self.a)
        self.prb.setDynamics(self.xdot)

        # underactuation constraints
        if self.contacts:
            id_fn = kin_dyn.InverseDynamics(self.kd, self.contacts, self.kd_frame)
            tau = id_fn.call(self.q, self.v, self.a, self.fmap)
            self.prb.createIntermediateConstraint('dynamics', tau[:6])
        # else:
        #     id_fn = kin_dyn.InverseDynamics(self.kd)


    # def getInput(self):
    #     return self.a
    #
    # def getState(self):
    #     return

class TaskInterface:
    def __init__(self,
                 urdf,
                 q_init: Dict[str, float],
                 base_init: np.array,
                 problem_opts: Dict[str, any],
                 model_description: str,
                 fixed_joints: List[str] = None,
                 contacts: List[str] = None): # todo this is wrong, it should not be listed in the initialization

        # get the model
        self.urdf = urdf.replace('continuous', 'revolute')
        self.fixed_joints = [] if fixed_joints is None else fixed_joints.copy()
        self.fixed_joints_pos = [q_init[k] for k in self.fixed_joints]
        fixed_joint_map = {k: q_init[k] for k in self.fixed_joints}
        self.kd = pycasadi_kin_dyn.CasadiKinDyn(self.urdf, fixed_joints=fixed_joint_map)
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

        self.joint_names = self.kd.joint_names()[2:]

        # number of dof
        self.nq = self.kd.nq()
        self.nv = self.kd.nv()

        # manage starting position
        # initial guess (also initial condition and nominal pose)
        q_init = {k: v for k, v in q_init.items() if k not in self.fixed_joints}
        self.q0 = self.kd.mapToQ(q_init)
        self.q0[:7] = base_init
        self.v0 = np.zeros(self.nv)
        # self.a0 = np.zeros(self.nv)

        # task list
        self.task_list = []

        self._createProblem(problem_opts)

        # model specification (whole body, centroidal, acceleration, velocity...)
        self.model = ModelDescription(self.prb, self.kd)
        self.model.generateModel(model_description)

        self.contacts = contacts
        for c in self.contacts:
            self.model.setContactFrame(c)

        # todo model is not initialized correctly!!!!!!!!!!
        # todo: add all the contacts
        self._initializeModel()

    def _createProblem(self, problem_opts):

        # definition of the problem
        self.N = problem_opts.get('N', 50)
        self.tf = problem_opts.get('tf', 10.0)
        self.dt = self.tf / self.N

        self.prb = Problem(self.N, receding=True)
        self.prb.setDt(self.dt)


    def _initializeModel(self):
        self.model.setDynamics()

    # a possible method could read from yaml and create the task list
    # here I do it manually

    def setTask(self, task_description):

        task_type = task_description['type']
        task_name = task_description['name']
        task_frame = None if 'frame' not in task_description else task_description['frame']
        task_indices = [0, 1, 2] if 'indices' not in task_description else task_description['indices'] # todo this is wrong
        task_nodes = [] if 'nodes' not in task_description else task_description['nodes']
        task_fun_type = None if 'fun_type' not in task_description else task_description['fun_type']
        task_weight = 1.0 if 'weight' not in task_description else task_description['weight']

        cartesian_type = 'position' if 'cartesian_type' not in task_description else task_description['cartesian_type']
        bound_scaling = 1. if 'bound_scaling' not in task_description else task_description['bound_scaling']

        if task_type == 'Cartesian':
            task = CartesianTask(task_name, self.prb, self.kd, task_frame, task_nodes, task_indices, task_weight, cartesian_type=cartesian_type, fun_type=task_fun_type)
        elif task_type == 'Force':
            # todo this generates another variable (f_c) for the frame of the contact: when to initialize the model?
            self.model.setContactFrame(task_frame)
            task = InteractionTask(task_name, self.prb, self.kd, task_frame, task_nodes, task_indices, weight=task_weight)
        elif task_type == 'Contact':
            task = ContactTask(task_name, self.prb, self.kd, task_frame, self.prb.getVariables('f_' + task_frame), task_nodes)
        elif task_type == 'Postural':
            task = PosturalTask(task_name, self.prb, self.kd, task_frame, task_nodes, task_indices, task_weight, fun_type=task_fun_type, postural_ref=self.q0[task_indices])
        elif task_type == 'JointLimits':
            task = JointLimitsTask(task_name, self.prb, self.kd, task_frame, task_nodes, task_indices, task_weight, fun_type=task_fun_type, bound_scaling=bound_scaling)
        else:
            raise Exception('Unknown task type {}'.format(task_type))

        self.task_list.append(task)

    def addTask(self, task):
        self.task_list.append(task)

    def getTask(self, task_name):
        list_1 = [t for t in self.task_list if t.getName() == task_name][0]
        return list_1

    # todo
    def setTaskOptions(self):
        pass