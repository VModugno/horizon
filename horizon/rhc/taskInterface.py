from horizon.problem import Problem
from horizon.utils import utils, kin_dyn
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.rhc.tasks.posturalTask import PosturalTask
from horizon.rhc.tasks.limitsTask import JointLimitsTask
from horizon.rhc.tasks.regularizationTask import RegularizationTask
from typing import List, Dict
import numpy as np
from horizon.rhc import task_factory, plugin_handler
from horizon.rhc.yaml_handler import YamlParser


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

    def setContactFrame(self, contact):

        # todo add more guards
        if contact in self.contacts:
            raise Exception(f'{contact} frame is already a contact.')
        self.contacts.append(contact)
        f_c = self.prb.createInputVariable('f_' + contact, self.nf)
        self.fmap[contact] = f_c
        return f_c

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

    def getContacts(self):
        return self.contacts
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

        # here I register the the default tasks
        # todo: should I do it here?
        task_factory.register('Cartesian', CartesianTask)
        task_factory.register('Force', InteractionTask)
        task_factory.register('Postural', PosturalTask)
        task_factory.register('JointLimits', JointLimitsTask)
        task_factory.register('Regularization', RegularizationTask)


        self.urdf = urdf.replace('continuous', 'revolute')
        self.fixed_joints = [] if fixed_joints is None else fixed_joints.copy()
        self.fixed_joints_pos = [q_init[k] for k in self.fixed_joints]
        fixed_joint_map = {k: q_init[k] for k in self.fixed_joints}
        self.kd = pycasadi_kin_dyn.CasadiKinDyn(self.urdf, fixed_joints=fixed_joint_map)
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

        self.joint_names = self.kd.joint_names()[2:]

        self.init_contacts = contacts

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

        if self.init_contacts is not None:
            for c in self.init_contacts:
                self.model.setContactFrame(c)

        self._initializeModel()

    def _createProblem(self, problem_opts):

        # definition of the problem
        # todo: what if I want a variable dt?
        self.N = problem_opts.get('N', 50)
        self.tf = problem_opts.get('tf', 10.0)
        self.dt = self.tf / self.N

        self.prb = Problem(self.N, receding=True)
        self.prb.setDt(self.dt)


    def _initializeModel(self):
        self.model.setDynamics()

    # a possible method could read from yaml and create the task list
    def setTaskFromYaml(self, task_config):

        # todo this should probably go in each single task definition --> i don't have the info from the ti then
        shortcuts = {
            'nodes': {'final': self.N, 'all': range(self.N + 1)},
            # todo: how to choose the value to substitute depending on the item? (indices of q: self.model.nq, indices of f: self.f.size ...)
            # 'indices': {'floating_base': range(7), 'joints': range(7, self.model.nq + 1)}
        }

        task_list = YamlParser.load(task_config)

        # todo: this should be updated everytime a task is added
        for task_descr in task_list:
            task_descr_resolved = YamlParser.resolve(task_descr, shortcuts)


            if 'weight' in task_descr and isinstance(task_descr['weight'], dict):
                weight_dict = task_descr['weight']
                if 'position' in weight_dict:
                    weight_dict['q'] = weight_dict.pop('position')
                if 'velocity' in weight_dict:
                    weight_dict['v'] = weight_dict.pop('velocity')
                if 'acceleration' in weight_dict:
                    weight_dict['a'] = weight_dict.pop('acceleration')

                # todo this is wrong: if new forces are added, this is not adding them into the Task
                if 'force' in weight_dict:
                    weight_force = weight_dict.pop('force')
                    for f in self.model.fmap.values():
                        weight_dict[f.getName()] = weight_force

            self.setTaskFromDict(task_descr_resolved)

        # tasks = [task_factory.create(self.prb, self.kd, task_description) for task_description in task_yaml]

    # here I do it manually
    def setTaskFromDict(self, task_description):

        task_specific = self.generateTaskContext(task_description)
        task = task_factory.create(task_specific)
        self.task_list.append(task)

        return task

    def generateTaskContext(self, task_description):
        '''
        add specific context to task depending on its type
        '''

        task_description_mod = task_description.copy()

        # add generic context
        task_description_mod['prb'] = self.prb
        task_description_mod['kin_dyn'] = self.kd

        # check for subtasks:
        subtask_list = task_description_mod.pop('subtask') if 'subtask' in task_description else []

        # search the subtask:
        subtask_dict = dict()
        for subtask_description in subtask_list:

            # child inherit from parent the values, if not present
            # parent define the context for the child: child can override it
            for key, value in task_description_mod.items():
                if key not in subtask_description:
                    subtask_description[key] = value

            subtask_description = self.generateTaskContext(subtask_description)
            # override factory and pass directly to parent all the arguments for the child
            subtask_type = subtask_description.pop('type')
            subtask_dict[subtask_type] = subtask_description

        task_description_mod.update(subtask_dict)

        # automatically provided info:
        if task_description_mod['type'] == 'Postural':
            task_description_mod['postural_ref'] = self.q0

        if task_description_mod['type'] == 'Force':
            contact_frame = task_description_mod['frame']
            if contact_frame not in self.model.contacts:
                self.model.setContactFrame(contact_frame)

            task_description_mod['force'] = self.prb.getVariables('f_' + task_description_mod['frame'])

        return task_description_mod

    def setTask(self, task):
        # check if task is of registered_type # todo what about plugins?
        assert isinstance(task, tuple(task_factory.get_registered_tasks()))
        self.task_list.append(task)

    def getTask(self, task_name):
        list_1 = [t for t in self.task_list if t.getName() == task_name][0]
        return list_1

    def loadPlugins(self, plugins):
        plugin_handler.load_plugins(plugins)

    def getTasksType(self, task_type=None):
        return task_factory.get_registered_tasks(task_type)

    # todo
    def setTaskOptions(self):
        pass