import casadi as cs
from urllib3 import Retry

from horizon.problem import Problem
from horizon.utils import utils, kin_dyn
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.contactTask import ContactTask
from horizon.rhc.tasks.interactionTask import InteractionTask, SurfaceContact, VertexContact
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.rhc.tasks.posturalTask import PosturalTask
from horizon.rhc.tasks.limitsTask import JointLimitsTask
from horizon.rhc.tasks.regularizationTask import RegularizationTask
from typing import List, Dict, Union, Tuple
import numpy as np
from horizon.rhc import task_factory, plugin_handler, solver_interface
from horizon.rhc.yaml_handler import YamlParser
from horizon.solvers.solver import Solver
import logging
import time

class ModelDescription:
    
    def __init__(self, problem, model):
        self.prb: Problem = problem
        self.kd = model

    def fk(self, frame) -> Tuple[Union[cs.SX, cs.MX]]:
        """
        returns the tuple (ee_pos, ee_rot), evaluated
        at the symbolic state variable q
        """
        fk_fn = self.kd.fk(frame)
        return fk_fn(self.q)

    def generateModel(self, model_type=None, floating_base=False):

        self.floating_base = floating_base
        self.model_type = 'whole_body' if model_type is None else model_type

        # todo choose
        if self.model_type == 'whole_body':

            self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

            self.nq = self.kd.nq()
            self.nv = self.kd.nv()

            # custom choices
            # todo this is ugly
            self.q = self.prb.createStateVariable('q', self.nq)
            self.v = self.prb.createStateVariable('v', self.nv)
            self.a = self.prb.createInputVariable('a', self.nv)

        else:
            raise NotImplementedError()


        self.fmap = dict()
        self.cmap = dict()

    def setContactFrame(self, contact_frame, contact_type, contact_params=dict()):

        # todo add more guards
        if contact_frame in self.fmap.keys():
            raise Exception(f'{contact_frame} frame is already a contact')

        if contact_type == 'surface':
            return self._make_surface_contact(contact_frame, contact_params)
        elif contact_type == 'vertex':
            return self._make_vertex_contact(contact_frame, contact_params) 
        elif contact_type == 'point':
            return self._make_point_contact(contact_frame, contact_params) 

        raise ValueError(f'{contact_type} is not a valid contact type')

    def _make_surface_contact(self, contact_frame, contact_params):
        # create input (todo: support degree > 0)
        wrench = self.prb.createInputVariable('f_' + contact_frame, dim=6)
        self.fmap[contact_frame] = wrench
        self.cmap[contact_frame] = [wrench]
        return wrench

    def _make_point_contact(self, contact_frame, contact_params):
        # create input (todo: support degree > 0)
        force = self.prb.createInputVariable('f_' + contact_frame, dim=3)
        self.fmap[contact_frame] = force
        self.cmap[contact_frame] = [force]
        return force

    def _make_vertex_contact(self, contact_frame, contact_params):
        
        vertex_frames = contact_params['vertex_frames']  # todo improve error

        # create inputs (todo: support degree > 0)
        vertex_forces = [self.prb.createInputVariable('f_' + vf, dim=3) for vf in vertex_frames]

        # save vertices
        for frame, force in zip(vertex_frames, vertex_forces):
            self.fmap[frame] = force

        self.cmap[contact_frame] = vertex_forces

        # do we need to reconstruct the total wrench?
        return vertex_forces

    def setDynamics(self):
        # todo refactor this floating base stuff
        if self.floating_base:
            self.xdot = utils.double_integrator_with_floating_base(self.q, self.v, self.a)
        else:
            self.xdot = utils.double_integrator(self.v, self.a)

        self.prb.setDynamics(self.xdot)

        # underactuation constraints
        if self.fmap:
            id_fn = kin_dyn.InverseDynamics(self.kd, self.fmap.keys(), self.kd_frame)
            self.tau = id_fn.call(self.q, self.v, self.a, self.fmap)
            self.prb.createIntermediateConstraint('dynamics', self.tau[:6])
        # else:
        #     id_fn = kin_dyn.InverseDynamics(self.kd)

    def getContacts(self):
        return self.cmap.keys()

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
                 is_receding: bool = True):

        # get the model

        self.is_receding = is_receding
        # here I register the the default tasks
        # todo: should I do it here?
        task_factory.register('Cartesian', CartesianTask)
        task_factory.register('Contact', ContactTask)
        task_factory.register('Wrench', SurfaceContact)
        task_factory.register('VertexForce', VertexContact)
        task_factory.register('Postural', PosturalTask)
        task_factory.register('JointLimits', JointLimitsTask)
        task_factory.register('Regularization', RegularizationTask)

        self.urdf = urdf.replace('continuous', 'revolute')
        self.fixed_joints = [] if fixed_joints is None else fixed_joints.copy()
        self.fixed_joints_pos = [q_init[k] for k in self.fixed_joints]
        fixed_joint_map = {k: q_init[k] for k in self.fixed_joints}
        self.kd = pycasadi_kin_dyn.CasadiKinDyn(self.urdf, fixed_joints=fixed_joint_map)
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

        # number of dof
        self.nq = self.kd.nq()
        self.nv = self.kd.nv()

        # manage starting position
        # initial guess (also initial condition and nominal pose)
        q_init = {k: v for k, v in q_init.items() if k not in self.fixed_joints}

        self.q0 = self.kd.mapToQ(q_init)

        floating_base = False
        if base_init is not None:
            self.q0[:7] = base_init
            floating_base = True
            self.joint_names = self.kd.joint_names()[2:]
        else:
            self.joint_names = self.kd.joint_names()[1:]

        self.v0 = np.zeros(self.nv)
        # self.a0 = np.zeros(self.nv)

        # task list
        self.task_list = []

        self._createProblem(problem_opts)

        # model specification (whole body, centroidal, acceleration, velocity...)
        self.model = ModelDescription(self.prb, self.kd)
        self.model.generateModel(model_description, floating_base=floating_base)

        

    def finalize(self):
        """
        to be called after all variables have been created
        """
        self.model.setDynamics()
        self.getSolver()

    
    def bootstrap(self):
        t = time.time()
        self.solver_bs.solve()
        elapsed = time.time() - t
        print(f'bootstrap solved in {elapsed} s')
        try:
            self.solver_bs.print_timings()
        except:
            pass
        self.solution = self.solver_bs.getSolutionDict()


    def save_solution(self, filename):
        from horizon.utils import mat_storer
        ms = mat_storer.matStorer(filename)
        self.solution['joint_names'] = self.joint_names
        ms.store(self.solution)


    def load_solution(self, filename):
        from horizon.utils import mat_storer
        ms = mat_storer.matStorer(filename)
        ig = ms.load()
        self.update_initial_guess(dk=0, from_dict=ig)


    def _createProblem(self, problem_opts):

        # definition of the problem
        # todo: what if I want a variable dt?
        self.N = problem_opts.get('ns', 50)
        self.tf = problem_opts.get('tf', 10.0)
        self.dt = self.tf / self.N

        self.prb = Problem(self.N, receding=self.is_receding, casadi_type=cs.MX, logging_level=logging.DEBUG)
        self.prb.setDt(self.dt)

    # a possible method could read from yaml and create the task list
    def setTaskFromYaml(self, yaml_config):

        self.task_desrc_list, self.non_active_task, self.solver_options = YamlParser.load(yaml_config)
        self.setSolverOptions(self.solver_options)

        # todo: this should be updated everytime a task is added
        for task_descr in self.task_desrc_list:

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

            self.setTaskFromDict(task_descr)


    def setTaskFromDict(self, task_description):
        # todo if task is dict... ducktyping

        task = self.generateTaskFromDict(task_description)
        self.setTask(task)
        return task

    def generateTaskFromDict(self, task_description):

        # todo this should probably go in each single task definition --> i don't have the info from the ti then
        shortcuts = {
            'nodes': {'final': self.N, 'all': range(self.N + 1)},
            # todo: how to choose the value to substitute depending on the item? (indices of q: self.model.nq, indices of f: self.f.size ...)
            # 'indices': {'floating_base': range(7), 'joints': range(7, self.model.nq + 1)}
        }
        task_descr_resolved = YamlParser.resolve(task_description, shortcuts)

        task_description_with_subtasks = self._handle_subtask(task_descr_resolved)
        task_specific = self.generateTaskContext(task_description_with_subtasks)

        task = task_factory.create(task_specific)

        self.setTask(task)

        return task

    def generateTaskContext(self, task_description):
        '''
        add specific context to task depending on its type
        '''

        task_description_mod = task_description.copy()
        # automatically provided info:

        # add generic context
        task_description_mod['prb'] = self.prb
        task_description_mod['kin_dyn'] = self.kd
        task_description_mod['model'] = self.model

        # add specific context
        if task_description_mod['type'] == 'Postural':
            task_description_mod['postural_ref'] = self.q0

        if task_description_mod['type'] == 'TorqueLimits':
            task_description_mod['var'] = self.model.tau

        return task_description_mod

    def _handle_subtask(self, task_description):

        # transform description of subtask (dict) into an instance of the task and pass it to the parent task
        task_description_copy = task_description.copy()
        # check for subtasks:
        subtasks = dict()
        if 'subtask' in task_description_copy:
            subtask_description_list = task_description_copy.pop(
                'subtask') if 'subtask' in task_description_copy else []


            # inherit from parent:
            for subtask_description in subtask_description_list:

                # TODO: wrong way to handle YAML subtasks
                if isinstance(subtask_description, str):
                    for task in self.non_active_task:
                        if task['name'] == subtask_description:
                            subtask_description = task
                            break


                # child inherit from parent the values, if not present
                # parent define the context for the child: child can override it
                
                # todo: better handling of parameter propagation
                # for key, value in task_description_copy.items():
                #     if key not in subtask_description and key != 'subtask':
                #         subtask_description[key] = value

                s_t = self.getTask(subtask_description['name'])
                
                if s_t is None:
                    s_t = self.generateTaskFromDict(subtask_description)

                subtasks[s_t.name] = s_t
                task_description_copy.update({'subtask': subtasks})

        return task_description_copy

    def setTask(self, task):
        # check if task is of registered_type # todo what about plugins?
        assert isinstance(task, tuple(task_factory.get_registered_tasks()))
        self.task_list.append(task)

    def getTask(self, task_name):
        # return the task with name task_name
        for task in self.task_list:
            if task.name == task_name:
                return task
        return None

    def getTaskByClass(self, task_class):
        list_1 = [t for t in self.task_list if isinstance(t, task_class)]
        return list_1

    def loadPlugins(self, plugins):
        plugin_handler.load_plugins(plugins)

    def getTasksType(self, task_type=None):
        return task_factory.get_registered_tasks(task_type)

    # todo
    def setTaskOptions(self):
        pass

    def setSolverOptions(self, solver_options):
        solver_type = solver_options.pop('type')
        is_receding = solver_options.pop('receding', False)
        self.si = solver_interface.SolverInterface(solver_type, is_receding, solver_options)

    def getSolver(self):

        if self.si.type != 'ilqr':
            # todo get options from yaml
            th = Transcriptor.make_method('multiple_shooting', self.prb)

        # todo if receding is true ....
        self.solver_bs = Solver.make_solver(self.si.type, self.prb, self.si.opts)
        
        try:
            self.solver_bs.set_iteration_callback()
        except:
            pass

        scoped_opts_rti = self.si.opts.copy()
        scoped_opts_rti['ilqr.enable_line_search'] = False
        scoped_opts_rti['ilqr.max_iter'] = 4
        # solver_rti = Solver.make_solver(self.si.type, self.prb, scoped_opts_rti)

        return self.solver_bs, None
