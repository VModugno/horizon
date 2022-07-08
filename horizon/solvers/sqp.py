#try:
from .pysqp import SQPGaussNewtonSX
from horizon.solvers.pyilqr import IterativeLQR
#except ImportError:
#    print('failed to import pysqp extension; did you compile it?')
#    exit(1)

from .solver import Solver
from horizon.problem import Problem
from typing import Dict
import numpy as np
import casadi as cs
import itertools


class GNSQPSolver(Solver):

    def __init__(self, prb: Problem, opts: Dict, qp_solver_plugin: str) -> None:

        super().__init__(prb, opts=opts)

        self.prb = prb

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        # generate problem to be solver
        var_list = list()
        for var in prb.var_container.getVarList(offset=False):
            var_list.append(var.getImpl())
        w = cs.veccat(*var_list)  #

        fun_list = list()
        for fun in prb.function_container.getCnstr().values():
            fun_list.append(fun.getImpl())
        g = cs.veccat(*fun_list)

        # build cost functions list
        cost_list = list()
        for fun in prb.function_container.getCost().values():
            cost_list.append(fun.getImpl())
        f = cs.vertcat(cs.veccat(*cost_list))

        # build parameters
        f_par_list = []
        for name in prb.function_container.getCost():
            f_par_list.append(prb.function_container.getCost()[name].getParameters())

        f_par_name_list = [pp.getName() for p in f_par_list for pp in p]
        f_input_str = ['x'] + f_par_name_list
        f_input = [w] + [pp.getImpl() for p in f_par_list for pp in p]

        g_par_list = []
        for name in prb.function_container.getCnstr():
            g_par_list.append(prb.function_container.getCnstr()[name].getParameters())

        g_par_name_list = [pp.getName() for p in g_par_list for pp in p]
        g_input_str = ['x'] + g_par_name_list
        g_input = [w] + [pp.getImpl() for p in g_par_list for pp in p]

        # create solver from prob
        F = cs.Function('f', f_input, [f], f_input_str, ['f'])
        G = cs.Function('g', g_input, [g], g_input_str, ['g'])

        self.solver = SQPGaussNewtonSX('gnsqp', qp_solver_plugin, F, G, self.opts)


    def set_iteration_callback(self, cb=None):
        if cb is None:
            self.solver.setIterationCallback(self._iter_callback)
        else:
            self.solver.setIterationCallback(cb)

    def _iter_callback(self, fpres):
            if not fpres.accepted:
                return
            fmt = ' <#09.3e'
            fmtf = ' <#04.2f'
            star = '*' if fpres.accepted else ' '
            fpres.print()

    def _set_param_values(self):
        params = self.prb.var_container.getParList()
        for p in params:
            self.solver.setParameterValue(p.getName(), cs.vertcat(*p.getValues()))

    def solve(self) -> bool:
        # update bounds and initial guess
        # update lower/upper bounds of variables
        lbw = self._getVarList('lb')
        ubw = self._getVarList('ub')
        # update initial guess of variables
        w0 = self._getVarList('ig')
        # update lower/upper bounds of constraints
        lbg = self._getFunList('lb')
        ubg = self._getFunList('ub')

        # update parameters
        self._set_param_values()

        # solve
        sol = self.solver.solve(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # self.cnstr_solution = self._createCnsrtSolDict(sol)

        # get solution dict
        self.var_solution = self._createVarSolDict(sol)

        # get solution as state/input
        self._createVarSolAsInOut(sol)

        # build dt_solution as an array
        self._createDtSol()

        return True

    def getSolutionDict(self):
        return self.var_solution

    def getConstraintSolutionDict(self):
        return self.cnstr_solution

    def getDt(self):
        return self.dt_solution

    def getHessianComputationTime(self):
        return self.solver.getHessianComputationTime()

    def getQPComputationTime(self):
        return self.solver.getQPComputationTime()

    def getLineSearchComputationTime(self):
        return self.solver.getLineSearchComputationTime()

    def getObjectiveIterations(self):
        return self.solver.getObjectiveIterations()

    def getConstraintNormIterations(self):
        return self.solver.getConstraintNormIterations()

    def setAlphaMin(self, alpha_min):
        self.solver.setAlphaMin(alpha_min)

    def getAlpha(self):
        return self.solver.getAlpha()

    def getBeta(self):
        return self.solver.getBeta()

    def setBeta(self, beta):
        self.solver.setBeta(beta)
