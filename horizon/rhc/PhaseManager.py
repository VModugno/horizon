import copy

import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs
from horizon import misc_function as misc
from horizon.problem import Problem
from horizon.utils import utils, kin_dyn
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers.solver import Solver
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.ros import replay_trajectory
from horizon.rhc.taskInterface import TaskInterface
import rospy
import os
import subprocess
import itertools
import time
import logging


class bcolors:

    CPURPLE = '\033[95m'
    CBLUE0 = '\033[94m'
    CCYAN0 = '\033[96m'
    CGREEN0 = '\033[92m'
    CYELLOW0 = '\033[93m'
    CRED0 = '\033[91m'

    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CUNDERLINE = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'


class Phase:
    def __init__(self, phase_name, n_nodes_phase):
        self.name = phase_name
        self.n_nodes = n_nodes_phase

        self.constraints = dict()
        self.costs = dict()

        self.vars = dict()
        self.vars_node = dict()
        self.var_bounds = dict()

        self.pars = dict()
        self.pars_node = dict()
        self.par_values = dict()

    def addConstraint(self, constraint, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.constraints[constraint] = active_nodes

    def addCost(self, cost, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.costs[cost] = active_nodes

    def addVariableBounds(self, var, lower_bounds, upper_bounds, nodes=None):
        # todo: this is not very nice, find another way? would be nice to copy the structure of addCost and addConstraint
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.vars[var.getName()] = var
        self.vars_node[var.getName()] = active_nodes
        self.var_bounds[var.getName()] = (lower_bounds, upper_bounds)

    def addParameterValues(self, par, values, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.pars[par.getName()] = par
        self.pars_node[par.getName()] = active_nodes
        self.par_values[par.getName()] = values

    def setDuration(self):
        pass


class PhaseToken(Phase):
    def __init__(self, *args, logger=None):
        self.__dict__ = args[0].__dict__.copy()

        if not logger:
            self.debug_mode = False
            self.logger = None
        else:
            self.logger = logger
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        self.id = '1337c0d3'
        # self.active_nodes = np.zeros(n_nodes_phase).astype(int)
        # self.active_nodes = [0] * n_nodes_phase
        # self.active_nodes = set()
        self.active_nodes = list()
        self.constraints_in_horizon = dict.fromkeys(self.constraints.keys(), set())
        self.costs_in_horizon = dict.fromkeys(self.costs.keys(), set())
        self.vars_in_horizon = dict.fromkeys(self.vars.keys(), set())
        self.pars_in_horizon = dict.fromkeys(self.pars.keys(), set())

    def update(self, initial_node):

        if self.debug_mode:
            self.logger.debug(f'{bcolors.CCYAN0}Phase "{self.name}" starting at node: {initial_node}{bcolors.CEND}')
            self.logger.debug(f'{bcolors.CCYAN0}Phase is active on nodes: {self.active_nodes}{bcolors.CEND}')

        # given the active nodes of a phase:
        #   for each constraint, get the active nodes and compute the horizon nodes it is active node
        for cnsrt, nodes in self.constraints.items():
            phase_nodes = [node for node in nodes if node in self.active_nodes]

            if cnsrt in self.constraints_in_horizon:
                if phase_nodes != self.constraints_in_horizon[cnsrt]:
                    if phase_nodes:
                        horizon_nodes = range(initial_node, initial_node + len(phase_nodes))
                        feas_nodes = [node for node in horizon_nodes if
                                      node in misc.getNodesFromBinary(cnsrt._getFeasNodes())]
                    else:
                        feas_nodes = []

                    self.constraints_in_horizon[cnsrt] = feas_nodes
                    if self.debug_mode:
                        self.logger.debug(
                            f'{bcolors.CCYAN0}   --->  {cnsrt.getName()}. Nodes to add: {list(feas_nodes)}{bcolors.CEND}')
                else:
                    del self.constraints_in_horizon[cnsrt]

        # ==============================================================================================================
        # ==============================================================================================================
        # ==============================================================================================================

        for cost, nodes in self.costs.items():
            phase_nodes = [node for node in nodes if node in self.active_nodes]

            if cost in self.costs_in_horizon:
                if phase_nodes != self.costs_in_horizon[cost]:
                    if phase_nodes:
                        horizon_nodes = range(initial_node, initial_node + len(phase_nodes))
                        feas_nodes = [node for node in horizon_nodes if
                                      node in misc.getNodesFromBinary(cost._getFeasNodes())]
                    else:
                        feas_nodes = []

                    self.costs_in_horizon[cost] = feas_nodes
                    if self.debug_mode:
                        self.logger.debug(
                            f'{bcolors.CCYAN0}   --->  {cost.getName()}. Nodes to add: {list(feas_nodes)}{bcolors.CEND}')
                else:
                    del self.costs_in_horizon[cost]

        # ==============================================================================================================
        # ==============================================================================================================
        # ==============================================================================================================

        for var, nodes in self.vars_node.items():
            phase_nodes = [node for node in nodes if node in self.active_nodes]

            if var in self.vars_in_horizon:
                if phase_nodes != self.vars_in_horizon[var]:
                    if phase_nodes:
                        horizon_nodes = range(initial_node, initial_node + len(phase_nodes))
                        feas_nodes = [node for node in horizon_nodes if
                                      node in misc.getNodesFromBinary(self.vars[var]._nodes_array)]
                    else:
                        feas_nodes = []

                    self.vars_in_horizon[var] = feas_nodes
                    if self.debug_mode:
                        self.logger.debug(
                            f'{bcolors.CCYAN0}   --->  {self.vars[var].getName()}. Nodes to add: {list(feas_nodes)}:{bcolors.CEND}')
                else:
                    del self.vars_in_horizon[var]

        # ==============================================================================================================
        # ==============================================================================================================
        # ==============================================================================================================
        for par, nodes in self.pars_node.items():
            phase_nodes = [node for node in nodes if node in self.active_nodes]

            if par in self.pars_in_horizon:
                if phase_nodes != self.pars_in_horizon[par]:
                    if phase_nodes:
                        horizon_nodes = range(initial_node, initial_node + len(phase_nodes))
                        feas_nodes = [node for node in horizon_nodes if
                                      node in misc.getNodesFromBinary(self.pars[par]._nodes_array)]
                    else:
                        feas_nodes = []

                    self.pars_in_horizon[par] = feas_nodes
                    if self.debug_mode:
                        self.logger.debug(
                            f'{bcolors.CCYAN0}   --->  {self.pars[par].getName()}. Nodes to add: {list(feas_nodes)}:{bcolors.CEND}')
                else:
                    del self.pars_in_horizon[par]

    def reset(self):
        self.active_nodes = list()
        self.constraints_in_horizon = self.constraints_in_horizon.fromkeys(self.constraints_in_horizon, set())
        self.costs_in_horizon = self.costs_in_horizon.fromkeys(self.costs_in_horizon, set())


"""
1. add a phase starting from a specific node
2. add a pattern that repeats (how to stop?)
3. 
"""


class HorizonManager:
    """
    this class is required to manage multiple phases update.
    If multiple phases use the same constraint, the constraint must be updated with the nodes from all the phases.
    (doing setNodes will reset all the nodes, so I need to keep track of the nodes (coming possibly from different phases) for the same constraint)
    """

    def __init__(self, logger=None):

        if not logger:
            self.debug_mode = False
            self.logger = None
        else:
            self.logger = logger
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        # self.phases = list()
        self.constraints = dict()
        self.costs = dict()

        self.vars = dict()  # dict of variableUpdaters
        self.pars = dict()  # dict of parameterUpdaters

    class VariableUpdater:
        def __init__(self, variable, logger=None):

            if not logger:
                self.debug_mode = False
                self.logger = None
            else:
                self.logger = logger
                self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

            self.name = variable.getName()
            self.var = variable

            self.active_nodes = list()
            self.horizon_nodes = list()
            self.lower_bounds = -np.inf * np.ones((self.var.getDim(), len(self.var.getNodes())))
            self.upper_bounds = np.inf * np.ones((self.var.getDim(), len(self.var.getNodes())))

        def update(self, nodes, bounds, phase_nodes, reset=False):
            # add new nodes and bounds to var if var already has them
            if reset:
                self.var.setBounds(self.lower_bounds, self.upper_bounds)

            self.active_nodes.extend(phase_nodes)
            self.horizon_nodes.extend(nodes)
            self.lower_bounds[:, nodes] = bounds[0][:, phase_nodes]
            self.upper_bounds[:, nodes] = bounds[1][:, phase_nodes]

            if not nodes:
                # if nodes is empty, resetting all the bounds
                bounds_mat_lb = -np.inf * np.ones((self.var.getDim(), len(self.var.getNodes())))
                bounds_mat_ub = np.inf * np.ones((self.var.getDim(), len(self.var.getNodes())))
                self.var.setBounds(bounds_mat_lb, bounds_mat_ub)
                if self.debug_mode:
                    # self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                    #                   f'updated variable {self.var.getName()}: {self.var.getLowerBounds()[0]}'
                    #                   f'{bcolors.CEND}')
                    self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                                      f'updated variable {self.var.getName()} at all nodes.'
                                      f'{bcolors.CEND}')
            else:
                # only fill the required nodes
                self.var.setBounds(self.lower_bounds[:, self.horizon_nodes], self.upper_bounds[:, self.horizon_nodes],
                                   self.horizon_nodes)
                if self.debug_mode:
                    # self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                    #                   f'updated variable {self.var.getName()}: {self.var.getLowerBounds()[0]}'
                    #                   f'{bcolors.CEND}')
                    self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                                      f'updated variable {self.var.getName()} at nodes: {self.horizon_nodes}'
                                      f'{bcolors.CEND}')

    class ParameterUpdater:
        def __init__(self, parameter, logger=None):

            if not logger:
                self.debug_mode = False
                self.logger = None
            else:
                self.logger = logger
                self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

            self.name = parameter.getName()
            self.par = parameter

            self.active_nodes = list()
            self.horizon_nodes = list()
            self.values = np.zeros((self.par.getDim(), len(self.par.getNodes())))

        def update(self, nodes, values, phase_nodes, reset=False):

            if reset:
                self.par.assign(self.values)

            self.active_nodes.extend(phase_nodes)
            self.horizon_nodes.extend(nodes)
            self.values[:, nodes] = values[:, phase_nodes]

            if not nodes:
                values_mat = np.zeros((self.par.getDim(), len(self.par.getNodes())))
                self.par.assign(values_mat)
                if self.debug_mode:
                    # self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} updated parameters {self.par.getName()}: {self.par.getValues()}{bcolors.CEND}')
                    self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} updated parameters {self.par.getName()}at all nodes.{bcolors.CEND}')
            else:
                self.par.assign(self.values[:, self.horizon_nodes], self.horizon_nodes)

                if self.debug_mode:
                    # self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} updated parameters {self.par.getName()}: {self.par.getValues()}{bcolors.CEND}')
                    self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} updated parameters {self.par.getName()} at nodes: {self.horizon_nodes}{bcolors.CEND}')

    # def add_phase(self, phase):
    #
    #     self.phases.append(phase)
    #     self.update_phase(phase)

    def update_function(self, container, fun, nodes):

        if fun not in container:
            container[fun] = set(nodes)
            # fun.setNodes(list(nodes))
            # if self.debug_mode:
            #     self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD}'f'Adding nodes to '
            #                       f'{bcolors.CUNDERLINE}new{bcolors.CEND}'
            #                       f'{bcolors.CCYAN0}{bcolors.CBOLD} function {fun.getName()}: {fun.getNodes()}{bcolors.CEND}')
        else:
            if set(nodes) != container[fun]:
                # todo: there is a small inefficiency: resetting all the nodes even if just only a part are added
                [container[fun].add(node) for node in nodes]
                # fun.setNodes(list(container[fun]))
                # if self.debug_mode:
                #     self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                #                       f'Adding nodes to function {fun.getName()}: {fun.getNodes()}{bcolors.CEND}')

    def update_variable(self, var, nodes, bounds, phase_nodes, reset=False):
        var_name = var.getName()
        if var_name not in self.vars:
            self.vars[var_name] = self.VariableUpdater(var, self.logger)

        for var_item in self.vars.values():
            var_item.update(nodes, bounds[var_name], phase_nodes, reset)  # update var with variable updater

    def update_parameter(self, par, nodes, values, phase_nodes, reset=False):
        par_name = par.getName()
        if par_name not in self.pars:
            self.pars[par_name] = self.ParameterUpdater(par, self.logger)

        for par_item in self.pars.values():
            par_item.update(nodes, values[par_name], phase_nodes, reset)  # update var with parameter updater

    def update_constraint(self, constraint, nodes):
        self.update_function(self.constraints, constraint, nodes)

    def update_cost(self, cost, nodes):
        self.update_function(self.costs, cost, nodes)

    def update_phase(self, phase, reset=False):
        # tic = time.time()

        # todo why functions does not require reset?
        # add bounds to constraints
        for constraint, nodes in phase.constraints_in_horizon.items():
            self.update_constraint(constraint, nodes)

        for cost, nodes in phase.costs_in_horizon.items():
            self.update_cost(cost, nodes)

        for var, nodes in phase.vars_in_horizon.items():
            self.update_variable(phase.vars[var], nodes, phase.var_bounds, phase.active_nodes, reset)

        for var, nodes in phase.pars_in_horizon.items():
            self.update_parameter(phase.pars[var], nodes, phase.par_values, phase.active_nodes, reset)

        # print('update_phase', time.time() - tic)
    def reset(self):

        self.constraints = dict()
        self.costs = dict()
        self.vars = dict()
        self.pars = dict()

    def set_horizon_nodes(self):
        # todo what about bounds in constraints?
        # todo incorporate these two
        for constraint, nodes in self.constraints.items():
            constraint.setNodes(list(nodes))
            if self.debug_mode:
                self.logger.debug(f'{bcolors.CBLUE}{bcolors.CUNDERLINE}{bcolors.CBOLD}'
                                  f'updated function {constraint.getName()}: {constraint.getNodes()}'
                                  f'{bcolors.CEND}')

        for cost, nodes in self.costs.items():
            cost.setNodes(list(nodes))
            if self.debug_mode:
                self.logger.debug(f'{bcolors.CBLUE}{bcolors.CUNDERLINE}{bcolors.CBOLD}'
                                  f'updated function {cost.getName()}: {cost.getNodes()}'
                                  f'{bcolors.CEND}')

        # for var, var_item in self.vars.items():
        #     var_item.var.setBounds(var_item.)
# def add_flatten_lists(the_lists):
#     result = []
#     for _list in the_lists:
#         result += _list
#     return result
#

class SinglePhaseManager:
    """
    set of actions which involves combinations of constraints and bounds
    """

    def __init__(self, nodes, name=None, logger=None):

        if not logger:
            self.debug_mode = False
            self.logger = None
        else:
            self.logger = logger
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        # prb: Problem, urdf, kindyn, contacts_map, default_foot_z
        self.name = name
        # self.prb = problem
        self.registered_phases = dict()  # container of all the registered phases
        self.n_tot = nodes  # self.prb.getNNodes() + 1

        self.horizon_manager = HorizonManager(self.logger)

        self.phases = list()
        self.active_phases = list()  # list of all the active phases
        self.activated_nodes = list()
        self.horizon_nodes = np.nan * np.ones(self.n_tot)

        # empty nodes in horizon --> at the very beginning, all
        self.trailing_empty_nodes = self.n_tot

        # self.default_action = Phase('default', 1)
        # self.registerPhase(self.default_action)

    def registerPhase(self, p):

        self.registered_phases[p.name] = p

    def getRegisteredPhase(self, p=None):

        if p is None:
            return list(self.registered_phases.items())
        else:
            return self.registered_phases[p] if p in self.registered_phases else None

    def addPhase(self, phases, pos=None):

        if self.debug_mode:
            self.logger.debug(
                f'{bcolors.CVIOLET} =========== Happening in timeline: {self.name} =========== {bcolors.CEND}')

        # stupid checks
        if isinstance(phases, list):
            for phase in phases:
                assert isinstance(phase, Phase)
        else:
            assert isinstance(phases, Phase)
            phases = [phases]

        if self.debug_mode:
            self.logger.debug(
                f'{bcolors.CRED}{bcolors.CBOLD} Adding phases: {[phase.name for phase in phases]} at position: {pos}{bcolors.CEND}')
        # print(f'{bcolors.FAIL} Current phases:')
        # for phase in self.phases:
        #     print(f'{bcolors.FAIL}    - {phase.name}: {phase.active_nodes}')

        # generate a phase token from each phase commanded by the user
        phases_to_add = [PhaseToken(phase) for phase in phases]
        # append or insert, depending on the user
        if pos is None:
            self.phases.extend(phases_to_add)
        else:
            # recompute empty nodes in horizon, clear active nodes of all the phases AFTER the one inserted
            # insert phase + everthing AFTER it back
            self.trailing_empty_nodes = self.n_tot - sum(len(s.active_nodes) for s in self.phases[:pos])
            # if I'm removing phases from the current stack, i need to remove current nodes from the phases
            [elem.reset() for elem in self.phases[pos:]]
            self.horizon_manager.reset()
            [self.horizon_manager.update_phase(phase) for phase in self.phases[:pos]]
            # todo this reset is wrong!
            # new active phases
            self.active_phases = [phase for phase in self.phases if phase.active_nodes]
            self.phases[pos:pos] = phases_to_add
            phases_to_add.extend(self.phases[pos + 1:])

        # print(f'{bcolors.CYELLOW} updating phases:')
        # for phase in phases_to_add:
        #     print(f'{bcolors.CYELLOW}    - {phase.name}{bcolors.CEND}')

        if self.trailing_empty_nodes > 0:
            # set active node for each added phase
            current_phase = 0
            for phase in phases_to_add:
                current_phase += 1
                self.pos_in_horizon = self.n_tot - self.trailing_empty_nodes
                self.trailing_empty_nodes -= phase.n_nodes
                len_phase = phase.n_nodes
                if self.trailing_empty_nodes <= 0:
                    remaining_nodes = len_phase + self.trailing_empty_nodes
                    phase.active_nodes.extend(range(remaining_nodes))
                    self.active_phases.append(phase)
                    phase.update(self.pos_in_horizon)
                    self.trailing_empty_nodes = 0
                    break
                self.active_phases.append(phase)
                phase.active_nodes.extend(range(len_phase))
                phase.update(self.pos_in_horizon)

            # update only if phase_to_add is inside horizon ( --> :current_phase)
            [self.horizon_manager.update_phase(phase) for phase in phases_to_add[:current_phase]]
            self.horizon_manager.set_horizon_nodes()

        # print(f'{bcolors.CRED} Updated phases:')
        # for phase in self.phases:
        #     print(f'{bcolors.CRED}    - {phase.name}: {phase.active_nodes}{bcolors.CEND}')

    def getActivePhase(self, phase=None):

        if phase is None:
            return self.active_phases.copy()
        else:
            raise NotImplementedError
            # return self.active_phases.[phase]

    def getHorizonNodes(self):

        assert self.horizon_nodes.shape[0] == self.n_tot
        return self.horizon_nodes

    def _shift_phases(self):

        if self.debug_mode:
            self.logger.debug(
                f'{bcolors.CVIOLET} =========== Happening in timeline: {self.name} =========== {bcolors.CEND}')
        # if phases is empty, skip everything
        if self.phases:
            self.active_phases = [phase for phase in self.phases if phase.active_nodes]


            last_active_phase = self.active_phases[-1]
            # if 'last active node' of 'last active phase' is the last of the phase, add new phase, otherwise continue to fill the phase
            if last_active_phase.n_nodes - 1 in last_active_phase.active_nodes:
                n_active_phases = len(self.active_phases)
                # add new active phase from list of added phases
                if n_active_phases < len(self.phases):
                    elem = self.phases[n_active_phases].active_nodes[-1] + 1 if self.phases[
                        n_active_phases].active_nodes else 0
                    self.phases[n_active_phases].active_nodes.append(elem)
            else:
                # add new node to last active phase
                elem = last_active_phase.active_nodes[-1] + 1 if last_active_phase.active_nodes else 0
                last_active_phase.active_nodes.append(elem)

            # remove first element in phases
            self.active_phases[0].active_nodes = self.active_phases[0].active_nodes[1:]

            # burn depleted phases
            if not self.phases[0].active_nodes:
                if self.debug_mode:
                    self.logger.debug(f'{bcolors.CYELLOW}Removing depleted phase: {self.phases[0].name}{bcolors.CEND}')
                self.phases = self.phases[1:]
            self.horizon_manager.reset()
            i = 0
            for phase in self.active_phases:
                phase.update(i)
                i += len(phase.active_nodes)

            [self.horizon_manager.update_phase(phase, reset=True) for phase in self.active_phases]
            self.horizon_manager.set_horizon_nodes()

        self.trailing_empty_nodes = self.n_tot - sum(len(s.active_nodes) for s in self.active_phases)

        # for phase in self.active_phases:
        #     phase.reset()

class PhaseManager:
    def __init__(self, nodes, opts=None):

        self.logger = None
        self.debug_mode = False

        if opts is not None and 'logging_level' in opts:
            self.logger = logging.getLogger('logger')
            self.logger.setLevel(level=opts['logging_level'])
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        self.nodes = nodes
        self.timelines = []
        self.n_timelines = 0

    def addTimeline(self, name=None):

        new_timeline = SinglePhaseManager(self.nodes, name, self.logger)
        self.timelines.append(new_timeline)
        self.n_timelines += 1

        return new_timeline

    def registerPhase(self, p, timeline):
        self.timelines[timeline].registerPhase(p)

    def addPhase(self, phase, pos=None, timeline=0):
        self.timelines[timeline].addPhase(phase, pos)

    def _shift_phases(self):

        for timeline in self.timelines:
            # tic = time.time()
            timeline._shift_phases()
            # print('timeline cycle:', time.time() - tic)


if __name__ == '__main__':

    n_nodes = 10
    prb = Problem(n_nodes, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    u = prb.createInputVariable('u', 2)
    # z = prb.createVariable('u', 2, nodes=[3, 4, 5])

    # z.getImpl([2, 3, 4])
    # exit()
    pm = PhaseManager(n_nodes)
    pm.addTimeline('0')

    # cnsrt4 = prb.createConstraint('constraint_4', x * z, nodes=[3, 4])
    cnsrt1 = prb.createIntermediateConstraint('constraint_1', x - u, [])
    cnsrt2 = prb.createConstraint('constraint_2', x - y, [])
    cnsrt3 = prb.createConstraint('constraint_3', 3 * x, [])
    cost1 = prb.createIntermediateResidual('cost_1', x + u, [])

    in_p = Phase('initial', 5)
    st_p = Phase('stance', 6)

    in_p.addConstraint(cnsrt1)
    in_p.addVariableBounds(y, np.array([[0, 0]] * 5).T, np.array([[0, 0]] * 5).T)
    st_p.addConstraint(cnsrt2)

    pm.registerPhase(in_p, 0)
    #
    print(pm.timelines)
    pm.addPhase(in_p, timeline=0)
    pm.addPhase(st_p, timeline=0)
    # pm.addPhase(in_p)
    #
    for j in range(15):
        print(f'--------- shifting {j} -----------')
        tic = time.time()
        pm._shift_phases()
        print('elapsed time in shifting:', time.time() - tic)

    exit()

    n_nodes = 11
    prb = Problem(n_nodes, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    u = prb.createInputVariable('u', 2)
    # z = prb.createVariable('u', 2, nodes=[3, 4, 5])

    # z.getImpl([2, 3, 4])
    # exit()
    pm = PhaseManager(n_nodes + 1)
    pm.addTimeline('0')
    pm.addTimeline('1')

    # cnsrt4 = prb.createConstraint('constraint_4', x * z, nodes=[3, 4])
    cnsrt1 = prb.createIntermediateConstraint('constraint_1', x - u, [])
    cnsrt2 = prb.createConstraint('constraint_2', x - y, [])
    cnsrt3 = prb.createConstraint('constraint_3', 3 * x, [])
    cost1 = prb.createIntermediateResidual('cost_1', x + u, [])

    in_p = Phase('initial', 5)
    st_p = Phase('stance', 6)
    fl_p = Phase('flight', 2)

    # in_p.addConstraint(cnsrt4)
    in_p.addConstraint(cnsrt1, nodes=range(0, 2))
    # in_p.addConstraint(cnsrt2)
    # in_p.addCost(cost1, nodes=[3, 4])
    fl_p.addConstraint(cnsrt2)
    st_p.addConstraint(cnsrt3)

    pm.registerPhase(in_p, timeline=0)
    pm.registerPhase(st_p, timeline=1)
    pm.registerPhase(fl_p, timeline=0)

    tic = time.time()
    for i in range(1):
        pm.addPhase(in_p, timeline=0)
        pm.addPhase(st_p, timeline=1)
        pm.addPhase(fl_p, timeline=0)
    print(f'elapsed_time: {time.time() - tic}')

    # for phase in pm.active_phases:
    #     print(f'{phase.name, phase.constraints}')
    # for c_name, c_item in prb.getConstraints().items():
    #     print(f'{c_name}: {c_item.getNodes()}')

    # print('========= all added phases ===========')
    # for phase in pm.phases:
    #     print(phase.name, phase.active_nodes)

    for j in range(30):
        print('--------- shifting -----------')
        tic = time.time()
        pm._shift_phases()
        print('elapsed time in shifting:', time.time() - tic)
        if j == 3:
            print('-------- adding phase -----------')
            pm.addPhase(in_p, 1)

        if j == 20:
            print('-------- adding phase -----------')
            pm.addPhase(st_p, 3)

        # for phase in pm.phases:
        #     print(f'{phase.name}:, {phase.active_nodes}:')

    # for phase in pm.active_phases:
    #     print(f'{phase.name, phase.constraints}')
    # for c_name, c_item in prb.getConstraints().items():
    #     print(f'{c_name}: {c_item.getNodes()}')
