import copy

import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs

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


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Phase:
    def __init__(self, phase_name, n_nodes_phase):
        self.name = phase_name
        self.n_nodes = n_nodes_phase

        self.constraints = dict()
        self.costs = dict()

    def addConstraint(self, constraint, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.constraints[constraint] = active_nodes

    def addCost(self, cost, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.costs[cost] = active_nodes

    def setDuration(self):
        pass


class PhaseToken(Phase):
    def __init__(self, *args):
        self.__dict__ = args[0].__dict__.copy()

        self.id = 'cazzi'
        # self.active_nodes = np.zeros(n_nodes_phase).astype(int)
        # self.active_nodes = [0] * n_nodes_phase
        # self.active_nodes = set()
        self.active_nodes = list()
        self.constraints_in_horizon = dict.fromkeys(self.constraints.keys(), set())
        self.costs_in_horizon = dict.fromkeys(self.costs.keys(), set())

    def update(self, initial_node, erasing=True):
        # [cnsrt.setNodes(nodes) for cnsrt, nodes in self.constraints.items()]
        # [cost.setNodes(nodes) for cost, nodes in self.costs.items()]

        for cnsrt, nodes in self.constraints.items():

            # self.constraints_in_horizon[cnsrt] = set([node + initial_node for node in nodes if node in self.active_nodes])
            cnsrt.setNodes([node + initial_node for node in nodes if node in self.active_nodes], erasing=erasing)

        for cost, nodes in self.costs.items():
            # self.costs_in_horizon[cost] = set([node + initial_node for node in nodes if node in self.active_nodes])
            cost.setNodes([node + initial_node for node in nodes if node in self.active_nodes], erasing=erasing)


# class PhaseContainer:
#     def __init__(self):
#         self.phases = list()
#         self.constraints = dict()
#         self.costs = dict()
#
#     def add_phase(self, phase):
#
#         self.phases.append(phase)
#
#         self.update_phase(phase)
#
#
#     def update_function(self, container, fun, nodes):
#
#         if fun not in container:
#             container[fun] = set(nodes)
#         else:
#             [container[fun].add(node) for node in nodes]
#
#         fun.setNodes(list(container[fun]))
#
#     def update_constraint(self, constraint, nodes):
#
#         self.update_function(self.constraints, constraint, nodes)
#
#     def update_cost(self, cost, nodes):
#
#         self.update_function(self.costs, cost, nodes)
#
#     def update_phase(self, phase):
#         for constraint, nodes in phase.constraints_in_horizon.items():
#             self.update_constraint(constraint, nodes)
#
#         for cost, nodes in phase.costs_in_horizon.items():
#             self.update_cost(cost, nodes)

"""
1. add a phase starting from a specific node
2. add a pattern that repeats (how to stop?)
3. 
"""


def add_flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    return result


class PhaseManager:
    """
    set of actions which involves combinations of constraints and bounds
    """

    def __init__(self, nodes, opts=None):
        # prb: Problem, urdf, kindyn, contacts_map, default_foot_z

        # self.prb = problem
        self.registered_phases = dict()  # container of all the registered phases
        self.phase_dict = dict()  # map of phase - identifier
        self.n_tot = nodes  # self.prb.getNNodes() + 1

        # empty nodes in horizon --> at the very beginning, all
        self.trailing_empty_nodes = self.n_tot

        self.phases = list()
        self.active_phases = list()  # list of all the active phases
        self.activated_nodes = list()
        self.horizon_nodes = np.nan * np.ones(self.n_tot)

        self.default_action = Phase('default', 1)
        self.registerPhase(self.default_action)

        # self.phase_container = PhaseContainer()

    def registerPhase(self, p):

        # todo: add unique number identifier?
        self.phase_dict[p.name] = len(self.registered_phases)
        self.registered_phases[p.name] = p

    def getRegisteredPhase(self, p=None):

        if p is None:
            return list(self.registered_phases.items())
        else:
            return self.registered_phases[p] if p in self.registered_phases else None

    def addPhase(self, phases, pos=None):

        # stupid checks
        if isinstance(phases, list):
            for phase in phases:
                assert isinstance(phase, Phase)
        else:
            assert isinstance(phases, Phase)
            phases = [phases]


        # generate a phase token from each phase commanded by the user
        # phases_to_add = [PhaseToken(phase.name, phase.n_nodes) for phase in phases]
        phases_to_add = [PhaseToken(phase) for phase in phases]
        # append or insert, depending on the user
        if pos is None:
            self.phases.extend(phases_to_add)
        else:
            # recompute empty nodes in horizon, clear active nodes of all the phases AFTER the one inserted
            # insert phase + everthing AFTER it back
            self.trailing_empty_nodes = self.n_tot - sum(len(s.active_nodes) for s in self.phases[:pos])
            [elem.active_nodes.clear() for elem in self.phases[pos:]]
            self.active_phases = [phase for phase in self.phases if phase.active_nodes]
            self.phases[pos:pos] = phases_to_add
            phases_to_add.extend(self.phases[pos + 1:])

        if self.trailing_empty_nodes > 0:
            # set active node for each added phase
            for phase in phases_to_add:
                self.pos_in_horizon = self.n_tot - self.trailing_empty_nodes
                self.trailing_empty_nodes -= phase.n_nodes
                len_phase = phase.n_nodes
                if self.trailing_empty_nodes <= 0:
                    remaining_nodes = len_phase + self.trailing_empty_nodes
                    phase.active_nodes.extend(range(remaining_nodes))
                    self.active_phases.append(phase)
                    phase.update(self.pos_in_horizon, erasing=False)
                    self.trailing_empty_nodes = 0
                    break
                self.active_phases.append(phase)
                phase.active_nodes.extend(range(len_phase))
                phase.update(self.pos_in_horizon, erasing=False)


        # [self.phase_container.update_phase(phase) for phase in phases_to_add]

        for phase in phases_to_add:
            print(f'{bcolors.OKBLUE}Adding Phase: {phase.name}')

        print('Current phases:')
        for phase in self.phases:
            print(f'{bcolors.FAIL} Phase: {phase.name}. N. nodes: {phase.n_nodes}. Active nodes: {phase.active_nodes}{bcolors.ENDC}')
            for constraint, def_nodes in phase.constraints.items():
                print(f'{bcolors.FAIL}         --->  {constraint.getName()} (defined on {list(def_nodes)}{bcolors.ENDC}: {bcolors.FAIL}{bcolors.BOLD} {constraint.getNodes()}{bcolors.ENDC}')
                # print(f'{bcolors.FAIL}         --->  {constraint.getName()} (defined on {list(def_nodes)}): {phase.constraints_in_horizon[constraint]}{bcolors.ENDC}')
        print(f'{bcolors.FAIL}-------------------------{bcolors.ENDC}')

    def getActivePhase(self, phase=None):

        if phase is None:
            return self.active_phases.copy()
        else:
            raise NotImplementedError
            # return self.active_phases.[phase]

    # def setPattern(self, pattern, n_start=None, n_stop=None):
    #
    #     for phase in pattern:
    #         assert isinstance(phase, Phase)
    #
    #     self.current_pattern = pattern
    #
    #     self._expand_phases_in_horizon(n_start, n_stop)

    # def _expand_phases_in_horizon(self, n_start=None, n_stop=None):
    #
    #     adds phases to future horizon (keep track of all the phases, also outside the problem horizon)
    # n_start = 0
    # n_stop = len(self.activated_nodes) if len(self.activated_nodes) <= self.n_tot else self.n_tot
    #
    # fill nodes in horizon (fixed number of nodes) with phases
    # self._update_horizon(n_start, n_stop)

    # def _update_horizon(self, n_start=None, n_stop=None):
    #
    #     n_start = 0 if n_start is None else n_start
    #     n_stop = self.n_tot if n_stop is None else n_stop
    #
    #     slice_node = slice(n_start, n_stop)
    #
    #     todo wrong implementation of slice nodes
    #     reset to default action
    # self.horizon_nodes[slice_node] = self.phase_dict['default']
    #
    # todo enhance
    # if slice_node has a bigger dimension than self.activated_nodes, what happens?
    # n_stop_activated = n_stop if n_stop < len(self.activated_nodes) else len(self.activated_nodes)
    # self.horizon_nodes[n_start:n_stop_activated] = self.activated_nodes[n_start:n_stop_activated]
    #
    # todo where to put?
    #
    # def _update_active_nodes(self, phases):
    #
    #     self.activated_nodes = []
    #     for phase in phases:
    #         phase_num = self.phase_dict[phase.name]
    #         self.activated_nodes.extend([phase_num] * self.registered_phases[phase.name].n_nodes)

    # def update_phases_from_active_nodes(self, active_nodes):
    #
    #     for phase in self.active_phase_list:
    #         if self.phase_dict[phase.name] not in active_nodes:
    #             self.active_phase_list.remove(phase)

    def getHorizonNodes(self):

        assert self.horizon_nodes.shape[0] == self.n_tot
        return self.horizon_nodes

    def _shift_phases(self):

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
                print(f'{bcolors.WARNING}Removing depleted phase: {self.phases[0].name}{bcolors.ENDC}')
                self.phases = self.phases[1:]

        self.trailing_empty_nodes = self.n_tot - sum(len(s.active_nodes) for s in self.active_phases)

        i = 0
        for phase in self.active_phases:
            phase.update(i, erasing=True)
            i += len(phase.active_nodes)
            # self.phase_container.update_phase(phase)

        for phase in self.phases:
            print(f'{bcolors.OKCYAN}Phase: {phase.name}. N. nodes: {phase.n_nodes}. Active nodes: {phase.active_nodes}{bcolors.ENDC}')
            for constraint, def_nodes in phase.constraints.items():
                print(f'{bcolors.OKCYAN}         --->  {constraint.getName()} (defined on {list(def_nodes)}){bcolors.ENDC}: {bcolors.OKCYAN}{bcolors.BOLD}{constraint.getNodes()}{bcolors.ENDC}')
                # print(f'{bcolors.OKCYAN}         --->  {constraint.getName()} (defined on {list(def_nodes)}){bcolors.ENDC}: {bcolors.OKCYAN}{bcolors.BOLD}{phase.constraints_in_horizon[constraint]}{bcolors.ENDC}')
            print(f'{bcolors.OKCYAN}-------------------------{bcolors.ENDC}')



if __name__ == '__main__':


    # n_nodes = 20
    # prb = Problem(n_nodes, receding=True, casadi_type=cs.SX)
    # x = prb.createStateVariable('x', 2)
    # y = prb.createStateVariable('y', 2)
    # u = prb.createInputVariable('u', 2)
    # # z = prb.createVariable('u', 2, nodes=[3, 4, 5])
    #
    # # z.getImpl([2, 3, 4])
    # # exit()
    # pm = PhaseManager(n_nodes)
    #
    # # cnsrt4 = prb.createConstraint('constraint_4', x * z, nodes=[3, 4])
    # cnsrt1 = prb.createIntermediateConstraint('constraint_1', x - u, [])
    # cnsrt2 = prb.createConstraint('constraint_2', x - y, [])
    # cnsrt3 = prb.createConstraint('constraint_3', 3 * x, [])
    # cost1 = prb.createIntermediateResidual('cost_1', x + u, [])
    #
    # in_p = Phase('initial', 5)
    # st_p = Phase('stance', 4)
    # fl_p = Phase('flight', 2)
    #
    # # in_p.addConstraint(cnsrt4)
    # in_p.addConstraint(cnsrt1, nodes=range(0, 2))
    #
    # # in_p.addConstraint(cnsrt2)
    # # in_p.addCost(cost1, nodes=[3, 4])
    # #
    # st_p.addConstraint(cnsrt3)
    #
    # pm.registerPhase(in_p)
    # pm.registerPhase(st_p)
    # pm.registerPhase(fl_p)
    #
    # pm.addPhase(in_p)
    # pm.addPhase(st_p)
    # pm.addPhase(in_p)
    #
    # exit()

    n_nodes = 11
    prb = Problem(n_nodes, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    u = prb.createInputVariable('u', 2)
    # z = prb.createVariable('u', 2, nodes=[3, 4, 5])

    # z.getImpl([2, 3, 4])
    # exit()
    pm = PhaseManager(n_nodes)

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
    #
    st_p.addConstraint(cnsrt3)

    pm.registerPhase(in_p)
    pm.registerPhase(st_p)
    pm.registerPhase(fl_p)

    tic = time.time()
    for i in range(1):
        pm.addPhase(in_p)
        pm.addPhase(st_p)
        pm.addPhase(fl_p)
    print(f'elapsed_time: {time.time() - tic}')

    for phase in pm.active_phases:
        print(f'{phase.name, phase.constraints}')
    for c_name, c_item in prb.getConstraints().items():
        print(f'{c_name}: {c_item.getNodes()}')

    print('========= all added phases ===========')
    for phase in pm.phases:
        print(phase.name, phase.active_nodes)

    for j in range(10):
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

    for phase in pm.active_phases:
        print(f'{phase.name, phase.constraints}')
    for c_name, c_item in prb.getConstraints().items():
        print(f'{c_name}: {c_item.getNodes()}')

    exit()
