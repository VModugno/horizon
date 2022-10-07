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
class Phase:
    def __init__(self, phase_name, n_nodes_phase):
        self.name = phase_name
        self.n_nodes = n_nodes_phase

    def setDuration(self):
        pass

class PhaseToken(Phase):
    def __init__(self, phase_name, n_nodes_phase):
        super().__init__(phase_name, n_nodes_phase)

        self.id = 'cazzi'
        # self.active_nodes = np.zeros(n_nodes_phase).astype(int)
        self.active_nodes = [] * n_nodes_phase


"""
1. add a phase starting from a specific node
2. add a pattern that repeats (how to stop?)
3. 
"""
class PhaseManager:
    """
    set of actions which involves combinations of constraints and bounds
    """

    def __init__(self, nodes, opts=None):
        # prb: Problem, urdf, kindyn, contacts_map, default_foot_z

        # self.prb = problem
        self.registered_phases = dict() # container of all the registered phases
        self.phase_dict = dict() # map of phase - identifier
        self.n_tot = nodes # self.prb.getNNodes() + 1

        self.phases = list()
        self.active_phase_list = list() # list of all the active phases
        self.activated_nodes = list()
        self.horizon_nodes = np.nan * np.ones(self.n_tot)

        self.default_action = Phase('default', 1)
        self.registerPhase(self.default_action)
        self.setPattern([self.default_action] * self.n_tot)

        # class HorizonNodes:
        #     def __init__(self, n_nodes):
        #         self.n_nodes = n_nodes
        #         self.horizon_nodes = []
        #
        #     def appendPhase(self, p):

    # def _set_default_action(self):
    #     pass
    # def init_constraints(self):
    #     pass
    #
    # def setContact(self, frame, nodes):
    #     pass
    #
    # def _append_nodes(self, node_list, new_nodes):
    #     for node in new_nodes:
    #         if node not in node_list:
    #             node_list.append(node)
    #
    #     return node_list
    #
    # def _append_params(self, params_array, new_params, nodes):
    #
    #     params_array[nodes] = np.append(params_array, new_params)
    #
    #     return params_array

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

        phases_to_add = [PhaseToken(phase.name, phase.n_nodes) for phase in phases]

        # append or insert
        if pos is None:
            self.phases.extend(phases_to_add)
        else:
            self.phases[pos:pos] = phases_to_add

        # active_nodes_list = np.concatenate([ap.active_nodes for ap in self.phases])
        active_nodes_list = []
        active_nodes_list[0:0] = [ap.active_nodes for ap in self.phases]

        print(active_nodes_list)
        exit()

        active_nodes_list[:] = [0]
        active_nodes_list[0:self.n_tot] = [1]


        i = 0
        for ap in self.phases:
            ap.active_nodes = active_nodes_list[i:i+ap.n_nodes]
            i += ap.n_nodes


        self.active_phase_list = [phase for phase in self.phases if np.isin(1, phase.active_nodes)]


        # self._update_active_nodes_from_phases(self.active_phase_list)
        # expand phases
        self._expand_phases_in_horizon()

    def getActivePhase(self, phase=None):

        if phase is None:
            return self.active_phase_list.copy()
        else:
            return self.active_phase_list[phase]

    def setPattern(self, pattern, n_start=None, n_stop=None):

        for phase in pattern:
            assert isinstance(phase, Phase)

        self.current_pattern = pattern

        self._expand_phases_in_horizon(n_start, n_stop)

    def _expand_phases_in_horizon(self, n_start=None, n_stop=None):

        # adds phases to future horizon (keep track of all the phases, also outside the problem horizon)
        n_start = 0
        n_stop = len(self.activated_nodes) if len(self.activated_nodes) <= self.n_tot else self.n_tot

        # fill nodes in horizon (fixed number of nodes) with phases
        self._update_horizon(n_start, n_stop)

    def _update_horizon(self, n_start=None, n_stop=None):

        n_start = 0 if n_start is None else n_start
        n_stop = self.n_tot if n_stop is None else n_stop

        slice_node = slice(n_start, n_stop)

        # todo wrong implementation of slice nodes
        # reset to default action
        self.horizon_nodes[slice_node] = self.phase_dict['default']

        # todo enhance
        #   if slice_node has a bigger dimension than self.activated_nodes, what happens?
        n_stop_activated = n_stop if n_stop < len(self.activated_nodes) else len(self.activated_nodes)
        self.horizon_nodes[n_start:n_stop_activated] = self.activated_nodes[n_start:n_stop_activated]

        # todo where to put?

    def _update_active_nodes(self, phases):

        self.activated_nodes = []
        for phase in phases:
            phase_num = self.phase_dict[phase.name]
            self.activated_nodes.extend([phase_num] * self.registered_phases[phase.name].n_nodes)

    # def update_phases_from_active_nodes(self, active_nodes):
    #
    #     for phase in self.active_phase_list:
    #         if self.phase_dict[phase.name] not in active_nodes:
    #             self.active_phase_list.remove(phase)

    def getHorizonNodes(self):

        assert self.horizon_nodes.shape[0] == self.n_tot
        return self.horizon_nodes

    def _shift_phases(self, shift_num):
        # burn the first 'shift_num' elements
        # phase_nodes = np.concatenate([ap.active_nodes for ap in self.phases])
        # phase_nodes = np.hstack((0, phase_nodes))
        phase_nodes = [ap.active_nodes for ap in self.phases]
        phase_nodes = phase_nodes.insert(0, 0)

        print(phase_nodes)
        exit()

        i = 0
        for ap in self.phases:
            ap.active_nodes = phase_nodes[i:i + ap.n_nodes]
            i += ap.n_nodes

        self.active_phase_list = [phase for phase in self.phases if 1 in phase.active_nodes]
        # self._update_horizon()

    # def execute(self, bootstrap_solution):
    #     """
    #     set the actions and spin
    #     """
    #     self._update_initial_state(bootstrap_solution, -1)
    # 
    #     self._set_default_action()
    #     k0 = 1
    # 
    #     for action in self.action_list:
    #         action.k_start = action.k_start - k0
    #         action.k_goal = action.k_goal - k0
    #         action_nodes = list(range(action.k_start, action.k_goal))
    #         action_nodes_in_horizon = [k for k in action_nodes if k >= 0]
    #         self._step(action)
    # 
    #     # for cnsrt_name, cnsrt in self.prb.getConstraints().items():
    #     #     print(cnsrt_name)
    #     #     print(cnsrt.getNodes().tolist())
    #     # remove expired actions
    #     self.action_list = [action for action in self.action_list if
    #                         len([k for k in list(range(action.k_start, action.k_goal)) if k >= 0]) != 0]
    #     # todo right now the non-active nodes of the parameter gets dirty,
    #     #  because .assing() only assign a value to the current nodes, the other are left with the old value
    #     #  better to reset?
    #     # self.pos_tgt.reset()
    #     # return 0
    # 
    #     ## todo should implement --> removeNodes()
    #     ## todo should implement a function to reset to default values
    # 
    # def _update_initial_state(self, bootstrap_solution, shift_num):
    # 
    #     x_opt = bootstrap_solution['x_opt']
    #     u_opt = bootstrap_solution['u_opt']
    # 
    #     xig = np.roll(x_opt, shift_num, axis=1)
    # 
    #     for i in range(abs(shift_num)):
    #         xig[:, -1 - i] = x_opt[:, -1]
    #     self.prb.getState().setInitialGuess(xig)
    # 
    #     uig = np.roll(u_opt, shift_num, axis=1)
    # 
    #     for i in range(abs(shift_num)):
    #         uig[:, -1 - i] = u_opt[:, -1]
    #     self.prb.getInput().setInitialGuess(uig)
    # 
    #     self.prb.setInitialState(x0=xig[:, 0])


if __name__ == '__main__':

    pm = PhaseManager(10)

    in_p = Phase('initial', 5)
    st_p = Phase('stance', 5)
    fl_p = Phase('flight', 2)

    pm.registerPhase(in_p)
    pm.registerPhase(st_p)
    pm.registerPhase(fl_p)

    phase_stance = pm.getRegisteredPhase('stance')

    print('starting horizon:', pm.getHorizonNodes())
    pm.addPhase(in_p)
    # print('added phase 1:', pm.getHorizonNodes())
    # print('phase list:', pm.getActivePhase())
    pm.addPhase(st_p)
    pm.addPhase(st_p)
    pm.addPhase(fl_p)

    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)

    print('-----------------------')
    for phase in pm.phases:
        print(phase.name, phase.active_nodes)
    # print('added phase 2:', pm.getHorizonNodes())
    # print('added phase 3 before phase 2:', pm.getHorizonNodes())
    # pattern = [in_p, st_p, in_p, fl_p]\
    tic = time.time()
    for i in range(1000):
        pm._shift_phases(1)
    toc = time.time() - tic
    print(toc)
    exit()
    print('--------- shifting --------------')
    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)

    pm._shift_phases(1)

    print('--------- shifting --------------')
    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)

    pm._shift_phases(1)

    print('--------- shifting --------------')
    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)


    pm._shift_phases(1)

    print('--------- shifting --------------')
    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)


    pm._shift_phases(1)
    print('--------- shifting --------------')
    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)

    pm._shift_phases(1)
    print('--------- shifting --------------')
    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)

    exit()
    pm._shift_phases(1)
    pm._shift_phases(1)
    pm._shift_phases(1)

    for phase in pm.phases:
        print(phase.name, phase.active_nodes)

    for phase in pm.active_phase_list:
        print(phase.name, phase.active_nodes)

    exit()
    # pm.setPattern(pattern)

    for i in range(10):
        pm._shift_phases(1)
        print('shift phases:', pm.getHorizonNodes())

        if i >= 9:
            print('adding phase "stance"')
            pm.addPhase(st_p)

    print('final situation:', pm.getHorizonNodes())
    print('phase list:', pm.getActivePhase())

    for i in range(6):
        pm._shift_phases(1)
        print('shift phases:', pm.getHorizonNodes())

