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

class Phase:
    def __init__(self, phase_name, n_nodes_phase):
        self.name = phase_name
        self.n_nodes = n_nodes_phase

    def setDuration(self):
        pass

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
        self.phase_container = dict()
        self.phase_dict = dict()
        self.n_tot = nodes # self.prb.getNNodes() + 1

        self.phase_list = list()
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
        self.phase_dict[p.name] = len(self.phase_container)
        self.phase_container[p.name] = p

    def getRegisteredPhase(self, p=None):

        if p is None:
            return list(self.phase_container.items())
        else:
            return self.phase_container[p] if p in self.phase_container else None

    def addPhase(self, phases, pos=None):

        # stupid checks
        if isinstance(phases, list):
            for phase in phases:
                assert isinstance(phase, Phase)
        else:
            assert isinstance(phases, Phase)
            phases = [phases]

        # append or insert
        if pos is None:
            self.phase_list.extend(phases)
        else:
            self.phase_list[pos:pos] = phases

        # expand phases
        self._expand_phases_in_horizon(self.phase_list)

    def getPhaseList(self):

        return self.phase_list.copy()

    def setPattern(self, pattern, n_start=None, n_stop=None):

        for phase in pattern:
            assert isinstance(phase, Phase)

        self.current_pattern = pattern

        self._expand_phases_in_horizon(pattern, n_start, n_stop)

    def _expand_phases_in_horizon(self, phases, n_start=None, n_stop=None):

        # adds phases to future horizon (keep track of all the phases, also outside the problem horizon)
        self.activated_nodes = []
        for phase in phases:
            phase_num = self.phase_dict[phase.name]
            self.activated_nodes.extend([phase_num] * self.phase_container[phase.name].n_nodes)

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
        self.horizon_nodes[slice_node] = self.activated_nodes[slice_node]

    def getHorizonNodes(self):

        assert self.horizon_nodes.shape[0] == self.n_tot
        return self.horizon_nodes

    def _shift_phases(self, shift_num):
        # burn the first 'shift_num' elements
        self.activated_nodes = self.activated_nodes[shift_num:]
        self._update_horizon()

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

    pm = PhaseManager(30)

    in_p = Phase('initial', 10)
    st_p = Phase('stance', 10)
    fl_p = Phase('flight', 10)

    pm.registerPhase(in_p)
    pm.registerPhase(st_p)
    pm.registerPhase(fl_p)

    phase_stance = pm.getRegisteredPhase('stance')

    print(pm.getHorizonNodes())
    pm.addPhase(in_p)
    print(pm.getHorizonNodes())
    print(pm.getPhaseList())
    pm.addPhase(st_p)
    print(pm.getHorizonNodes())
    pm.addPhase(fl_p, 1)
    print(pm.getHorizonNodes())
    # pattern = [in_p, st_p, in_p, fl_p]

    # pm.setPattern(pattern)


    for i in range(2):
        pm._shift_phases(1)
        print(pm.getHorizonNodes())
