import casadi as cs
from classes import function as fc
from classes import nodes as nd
from classes import state_variables as sv
import numpy as np
import logging
import sys
class Problem:

    def __init__(self, N, crash_if_suboptimal=False):

        self.logger = logging.getLogger('logger')
        # self.logger.setLevel(level=logging.DEBUG)

        stdout_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stdout_handler)

        self.crash_if_suboptimal = crash_if_suboptimal

        self.nodes = N + 1
        # state variable to optimize
        self.state_var_container = sv.StateVariables(self.nodes)

        # just variables
        self.var_container = list()
        # constraint variables
        self.cnstr_container = list()
        self.cnstr_impl = list()

        self.costfun_container = list()
        self.costfun_impl = list()

    def createStateVariable(self, name, dim, prev_nodes=None):
        var = self.state_var_container.setStateVar(name, dim, prev_nodes)
        return var

    def createInputVariable(self, name, dim, prev_nodes=None):
        var = self.state_var_container.setInputVar(name, dim, prev_nodes)
        return var

    # def setVariable(self, name, var):

        # assert (isinstance(var, (cs.casadi.SX, cs.casadi.MX)))
        # setattr(Problem, name, var)
        # self.var_container.append(name)

    # def getStateVariable(self, name):
    #
    #     for var in self.var_container:
    #         if var.getName() == name:
    #             return var
    #     return None

    def _getUsedVar(self, f):
        used_var = dict()

        for name_var, value_var in self.state_var_container.getVarAbstrDict().items():
            if cs.depends_on(f, value_var):
                used_var[name_var] = value_var

        return used_var

    # @classmethod
    # def createFunction(self, fun_type, **kwargs):
    #         return self.function[fun_type](**kwargs)

    def createConstraint(self, name, g, nodes=None, bounds=None):

        container = self.cnstr_container

        if not nodes:
            nodes = [0, self.nodes]

        used_var = self._getUsedVar(g)

        self.logger.debug('Creating function {}: {} with abstract variables {}'.format(name, g, used_var))
        fun = fc.Constraint(name, g, used_var, nodes, bounds)
        container.append(fun)

        return fun

    def createCostFunction(self, name, j, nodes=None):

        container = self.costfun_container
        if not nodes:
            nodes = [0, self.nodes]

        used_var = self._getUsedVar(j)

        fun = fc.CostFunction(name, j, used_var, nodes)
        container.append(fun)

        return fun

    def removeCostFunction(self, name):

        self.logger.debug('Functions before removal:', self.costfun_container)
        for fun in self.costfun_container:
            if fun.getName() == name:
                self.costfun_container.remove(fun)

        self.logger.debug('Function after removal:', self.costfun_container)

    def removeConstraint(self, name):
        for fun in self.cnstr_container:
            if fun.getName() == name:
                self.cnstr_container.remove(fun)

    def _implementFunctions(self, container, node):
        f_impl = list()

        # TODO be careful about ordering
        for fun in container:
            f = fun.getFunction()

            # implement constraint only if constraint is present in node k
            if node in fun.getNodes():
                used_vars = list()
                for name, val in fun.getVariables().items():
                    var = self.state_var_container.getVarImpl(name, node)
                    used_vars.append(var)

                f_impl.append(f(*used_vars))
                self.logger.debug('Implemented function "{}": {} with vars {}'.format(fun.getName(), f_impl, used_vars))
        return f_impl

    def _updateConstraints(self, node):
        temp_cnsrt_impl = self._implementFunctions(self.cnstr_container, node)
        if temp_cnsrt_impl:
            # add implemented constraints in list
            self.cnstr_impl += temp_cnsrt_impl


    # def getVariablesName(self):
    #     return [name for name, var in self.var]

    def _updateCostFunctions(self, node):

        temp_costfun_impl = self._implementFunctions(self.costfun_container, node)
        if temp_costfun_impl:
            # add implemented cost function in list
            self.costfun_impl += temp_costfun_impl

    # todo add setStateBoundsFromName
    # def setStateBoundsFromName(self, name, ubw, lbw, nodes=None):

    def createProblem(self):

        for k in range(self.nodes):  # todo decide if N or N+1
            self.logger.debug('Node {}:'.format(k))
            # implement the abstract state variable with the current node
            self.state_var_container.update(k)
            # implement the constraint
            self._updateConstraints(k) #todo not sure but ok, maybe better a constraint class container that updates takin state_var_container?
            self._updateCostFunctions(k)
            self.logger.debug('===========================================')

            self.costfun_sum = cs.sum1(cs.vertcat(*self.costfun_impl))


        # self.logger.debug('state var unraveled:', self.state_var_container.getVarImplList())
        # self.logger.debug('constraints unraveled:', cs.vertcat(*self.cnstr_impl))
        # self.logger.debug('cost functions unraveled:', cs.vertcat(*self.costfun_impl))
        # self.logger.debug('cost function summed:', self.costfun_sum)
        # self.logger.debug('----------------------------------------------------')

        j = self.costfun_sum
        w = self.state_var_container.getVarImplList()
        g = cs.vertcat(*self.cnstr_impl)
        self.prob = {'f': j, 'x': w, 'g': g}

    def solveProblem(self):

        self.state_var_container.updateBounds()
        self.state_var_container.updateInitialGuess()

        w = self.state_var_container.getVarImplList()
        w0 = self.state_var_container.getInitialGuessList()
        self.logger.debug('Initial guess vector for variables:'.format(self.state_var_container.getInitialGuessList()))

        g = cs.vertcat(*self.cnstr_impl)
        j = self.costfun_sum

        lbg = []
        ubg = []

        for node in range(self.nodes):
            for cnstr in self.cnstr_container:
                if node in cnstr.getNodes():
                    lbg += cnstr.getBoundsMin(node)
                    ubg += cnstr.getBoundsMax(node)

        lbw = self.state_var_container.getBoundsMinList()
        ubw = self.state_var_container.getBoundsMaxList()


        self.logger.debug('================')
        self.logger.debug('len w: {}'.format(w.shape))
        self.logger.debug('len lbw: {}'.format(len(lbw)))
        self.logger.debug('len ubw: {}'.format(len(ubw)))
        self.logger.debug('len w0: {}'.format(len(w0)))
        self.logger.debug('len g: {}'.format(g.shape))
        self.logger.debug('len lbg: {}'.format(len(lbg)))
        self.logger.debug('len ubg: {}'.format(len(ubg)))


        self.logger.debug('================')
        self.logger.debug('w: {}'.format(w))
        self.logger.debug('lbw: {}'.format(lbw))
        self.logger.debug('ubw: {}'.format(ubw))
        self.logger.debug('g: {}'.format(g))
        self.logger.debug('lbg: {}'.format(lbg))
        self.logger.debug('ubg: {}'.format(ubg))
        self.logger.debug('j: {}'.format(j))

        self.solver = cs.nlpsol('solver', 'ipopt', self.prob)#,
                           # {'ipopt': {'linear_solver': 'ma27', 'tol': 1e-4, 'print_level': 3, 'sb': 'yes'},
                           #  'print_time': 0})  # 'acceptable_tol': 1e-4(ma57) 'constr_viol_tol':1e-3

        sol = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        if self.crash_if_suboptimal:
            if not self.solver.stats()['success']:
                raise Exception('Optimal solution NOT found.')

        w_opt = sol['x'].full().flatten()

        # split solution for each variable
        abstract_vars = list(self.state_var_container.getVarAbstrDict(past=False).keys())
        solution_dict = {k: [] for k in abstract_vars}

        pos = 0

        for node, val in self.state_var_container.getVarImplDict().items():
            self.logger.debug('Node: {}'.format(node))

            for name, var in val.items():
                dim = var['var'].shape[0]
                sol = w_opt[pos:pos + dim]

                self.logger.debug('var {} of dim {}'.format(name, var['var'].shape[0]))
                # self.logger.debug('var {} of dim {}'.format(name, var['var'].shape[0]))
                self.logger.debug('Previous state: {}'.format(solution_dict))
                self.logger.debug('Var state: {}'.format(solution_dict[name]))
                self.logger.debug('Appending to {} opt sol [{}-{}]: {}'.format(name, pos, pos + dim, sol))
                solution_dict[name].extend(sol)
                self.logger.debug('Current state: {}'.format(solution_dict))
                self.logger.debug('~~~~~~~~~~~~~')
                pos = pos + dim

        return solution_dict

    # def getNode(self, n):

# Problem.function = {
#     'constraint': Problem.createConstraint,
#     'cost_function': Problem.createCostFunction,
# }
    # def setConstraint(self, cnstr):
    #     assert(isinstance(cnstr, fc.Constraint))
    #     self.cnstr_container.append(cnstr.getName())

if __name__ == '__main__':
    prb = Problem(10)