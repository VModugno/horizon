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

    def compute_node_in_horizon(self, initial_node):
        # [cnsrt.setNodes([node + initial_node for node in nodes if node in self.active_nodes], erasing=erasing) for cnsrt, nodes in self.constraints.items()]
        # [cost.setNodes([node + initial_node for node in nodes if node in self.active_nodes], erasing=erasing) for cost, nodes in self.costs.items()]
        # constraint_in_horizon = {}
        for cnsrt, nodes in self.constraints.items():
            print(f'{bcolors.OKCYAN}Phase is at position {initial_node}{bcolors.ENDC}')
            print(f'{bcolors.OKCYAN}Phase "{self.name}" defined on {list(nodes)} is active on nodes: {self.active_nodes}{bcolors.ENDC}')

            phase_nodes = [node for node in nodes if node in self.active_nodes]

            if cnsrt in self.constraints_in_horizon:
                if phase_nodes != self.constraints_in_horizon[cnsrt]:
                    if phase_nodes:
                        horizon_nodes = range(initial_node, initial_node + len(phase_nodes))
                        feas_nodes = [node for node in horizon_nodes if node in misc.getNodesFromBinary(cnsrt._getFeasNodes())]
                    else:
                        feas_nodes = []

                    self.constraints_in_horizon[cnsrt] = feas_nodes
                    print(f'{bcolors.OKCYAN}   --->  {cnsrt.getName()}. Nodes to add: {list(feas_nodes)}:{bcolors.ENDC}')
                else:
                    del self.constraints_in_horizon[cnsrt]

        # return constraints_in_horizon

    # def reset(self):
    #     self.constraints_in_horizon = dict()
    #     self.costs_in_horizon = dict()

"""
1. add a phase starting from a specific node
2. add a pattern that repeats (how to stop?)
3. 
"""

class PhaseContainer:
    """
    this class is required to manage multiple phases update.
    If multiple phases use the same constraint, the constraint must be updated with the nodes from all the phases.
    (doing setNodes will reset all the nodes, so I need to keep track of the nodes (coming possibly from different phases) for the same constraint)
    """
    def __init__(self):
        # self.phases = list()
        self.constraints = dict()
        self.costs = dict()

    # def add_phase(self, phase):
    #
    #     self.phases.append(phase)
    #     self.update_phase(phase)

    def update_function(self, container, fun, nodes):

        if fun not in container:
            container[fun] = set(nodes)
            fun.setNodes(list(nodes))
            print(f'{bcolors.OKCYAN}{bcolors.BOLD} updated function {fun.getName()}: {fun.getNodes()}{bcolors.ENDC}')
        else:
            if set(nodes) != container[fun]:
                # todo: there is a small inefficiency: resetting all the nodes even if just only a part are added
                [container[fun].add(node) for node in nodes]
                fun.setNodes(list(container[fun]))
                print(f'{bcolors.OKCYAN}{bcolors.BOLD} updated function {fun.getName()}: {fun.getNodes()}{bcolors.ENDC}')



    def update_constraint(self, constraint, nodes):
        self.update_function(self.constraints, constraint, nodes)

    def update_cost(self, cost, nodes):
        self.update_function(self.costs, cost, nodes)

    def update_phase(self, phase, initial_node):
        phase.compute_node_in_horizon(initial_node)


        for constraint, nodes in phase.constraints_in_horizon.items():
            self.update_constraint(constraint, nodes)

        # for cost, nodes in phase.costs_in_horizon.items():
        #     self.update_cost(cost, nodes)

    def reset(self):
        self.constraints = dict()
        self.costs = dict()

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

    def __init__(self, nodes, name=None):
        # prb: Problem, urdf, kindyn, contacts_map, default_foot_z
        self.name = name
        # self.prb = problem
        self.registered_phases = dict()  # container of all the registered phases
        self.n_tot = nodes  # self.prb.getNNodes() + 1

        self.phase_container = PhaseContainer()

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

        print(f'{bcolors.HEADER} =========== Happening in timeline: {self.name} =========== {bcolors.ENDC}')
        # stupid checks
        if isinstance(phases, list):
            for phase in phases:
                assert isinstance(phase, Phase)
        else:
            assert isinstance(phases, Phase)
            phases = [phases]


        # generate a phase token from each phase commanded by the user
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
                    # phase.update(self.pos_in_horizon)
                    self.trailing_empty_nodes = 0
                    break
                self.active_phases.append(phase)
                phase.active_nodes.extend(range(len_phase))
                # phase.update(self.pos_in_horizon)

        [self.phase_container.update_phase(phase, self.pos_in_horizon) for phase in phases_to_add]
        # print(f'{bcolors.HEADER} =========== Timeline: {self.name} =========== {bcolors.ENDC}')
        # for phase in phases_to_add:
        #     print(f'{bcolors.OKBLUE}Adding Phase: {phase.name}')
        #
        # print('Current phases:')
        # for phase in self.phases:
        #     print(f'{bcolors.FAIL} Phase: {phase.name}. N. nodes: {phase.n_nodes}. Active nodes: {phase.active_nodes}{bcolors.ENDC}')
        #     for constraint, def_nodes in phase.constraints.items():
        #         print(f'{bcolors.FAIL}         --->  {constraint.getName()} (defined on {list(def_nodes)}{bcolors.ENDC}: {bcolors.FAIL}{bcolors.BOLD} {constraint.getNodes()}{bcolors.ENDC}')
        #         print(f'{bcolors.FAIL}         --->  {constraint.getName()} (defined on {list(def_nodes)}): {phase.constraints_in_horizon[constraint]}{bcolors.ENDC}')
        # print(f'{bcolors.FAIL}-------------------------{bcolors.ENDC}')

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

        print(f'{bcolors.HEADER} =========== Happening in timeline: {self.name} =========== {bcolors.ENDC}')
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

            self.phase_container.reset()

            i = 0
            for phase in self.active_phases:
                # phase.update(i)
                self.phase_container.update_phase(phase, i)
                i += len(phase.active_nodes)



            # [self.phase_container.update_phase(phase) for phase in self.active_phases]

        self.trailing_empty_nodes = self.n_tot - sum(len(s.active_nodes) for s in self.active_phases)

        # for phase in self.active_phases:
        #     phase.reset()



        # print(f'{bcolors.HEADER} =========== Timeline: {self.name} =========== {bcolors.ENDC}')
        # for phase in self.phases:
        #     print(f'{bcolors.OKCYAN}Phase: {phase.name}. N. nodes: {phase.n_nodes}. Active nodes: {phase.active_nodes}{bcolors.ENDC}')
        #     for constraint, def_nodes in phase.constraints.items():
        #         print(f'{bcolors.OKCYAN}         --->  {constraint.getName()} (defined on {list(def_nodes)}){bcolors.ENDC}: {bcolors.OKCYAN}{bcolors.BOLD}{constraint.getNodes()}{bcolors.ENDC}')
        # print(f'{bcolors.OKCYAN}         --->  {constraint.getName()} (defined on {list(def_nodes)}){bcolors.ENDC}: {bcolors.OKCYAN}{bcolors.BOLD}{phase.constraints_in_horizon[constraint]}{bcolors.ENDC}')
        # print(f'{bcolors.OKCYAN}-------------------------{bcolors.ENDC}')


class PhaseManager:
    def __init__(self, nodes, opts=None):
        self.nodes = nodes
        self.timelines = []
        self.n_timelines = 0

    def addTimeline(self, name=None):

        self.timelines.append(SinglePhaseManager(self.nodes, name))
        self.n_timelines += 1
    def registerPhase(self, p, timeline):
        self.timelines[timeline].registerPhase(p)

    def addPhase(self, phase, pos=None, timeline=0):
        self.timelines[timeline].addPhase(phase, pos)

    def _shift_phases(self):

        for timeline in self.timelines:
            timeline._shift_phases()






if __name__ == '__main__':

    n_nodes = 6
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
    st_p.addConstraint(cnsrt2)

    pm.registerPhase(in_p, 0)
    #
    print(pm.timelines)
    pm.addPhase(in_p, timeline=0)
    pm.addPhase(in_p, timeline=0)
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
    pm = PhaseManager(n_nodes)
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