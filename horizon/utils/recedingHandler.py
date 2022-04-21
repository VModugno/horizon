import numpy

from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.variables import SingleVariable, SingleParameter
import numpy as np
import casadi as cs


class RecedingHandler:
    def __init__(self, solver: Solver):
        self.prb = solver.prb
        self.slv = solver

    def recede(self):

        self.slv.solve()
        solution = self.slv.getSolutionDict()

        self._recedeHorizon()

        self._updateInitialGuess(solution)

    def _recedeHorizon(self):

        for name_var, var in self.prb.getVariables().items():
            if isinstance(var, SingleVariable):
                pass
            else:
                var.shift()

        for name_par, par in self.prb.getParameters().items():
            if isinstance(par, SingleParameter):
                pass
            else:
                par.shift()

        for name_cnsrt, cnsrt in self.prb.getConstraints().items():
            if cnsrt.getName() == 'multiple_shooting':
                pass
            else:
                cnsrt.shift()

        # not necessary, because the shifting parameter, weight cost, is already shifted above
        # for name_cost, cost in self.prb.getCosts().items():
        #     cost.shift()

    def _updateInitialGuess(self, solution):

        for name_var, var in self.prb.getVariables().items():
            shifted_vals = solution[name_var][:, 1:]
            new_vals = np.zeros([var.getDim(), 1])

            var.setInitialGuess(np.hstack((shifted_vals, new_vals)))


if __name__ == '__main__':

    # N = 10
    # nodes_vec = np.array(range(N + 1))  # nodes = 10
    # dt = 0.01
    # prb = Problem(N, receding=True, casadi_type=cs.SX)
    # x = prb.createStateVariable('x', 2)
    # y = prb.createInputVariable('y', 2)
    # par = prb.createParameter('par', 2)
    # y_prev = y.getVarOffset(-1)
    # prb.setDynamics(x)
    # prb.setDt(dt)
    #
    # c = prb.createConstraint('c1', x[0], nodes=[3, 4, 5, 6])
    # cost = prb.createCost('cost', x, nodes=[2, 3, 4, 5])
    #
    # par.assign([1, 2], nodes=[3, 4, 5, 6, 7, 8])
    # print(par.getValues())
    # x.setUpperBounds([2, 3], nodes=[4, 5, 6])
    # x.setUpperBounds([1, 2], nodes=10)
    # print(x.getLowerBounds())
    # print(x.getUpperBounds())
    #
    # # p1 = prb.createParameter('p1', 4)
    # # cnsrt = prb.createConstraint('cnsrt', x - y_prev, nodes=[1])
    # # cnsrt = prb.createConstraint('cnsrt', x, nodes=[1])
    # # cost1 = prb.createCost('cost_x', 1e-3 * cs.sumsqr(x))
    # # cost2 = prb.createIntermediateCost('cost_y', 1e-3 * cs.sumsqr(y))
    #
    # c.setLowerBounds(-2)
    # print(c.getLowerBounds())
    # print(c.getUpperBounds())
    #
    # opts = dict()
    # opts['ipopt.tol'] = 1e-12
    # opts['ipopt.constr_viol_tol'] = 1e-12
    # slvr = Solver.make_solver('ipopt', prb, opts=opts)
    # # slvr.solve()
    # # sol = slvr.getSolutionDict()
    # #
    # # print('x:\n', sol['x'])
    # # print('y:\n', sol['y'])
    #
    # # print('before receding:')
    # # print('state bounds:')
    # # print(x.getLowerBounds())
    # # print(x.getUpperBounds())
    # # print('constraint bounds:')
    # # print(c.getLowerBounds())
    # # print(c.getUpperBounds())
    # print(x.getInitialGuess())
    # rcd = RecedingHandler(slvr)
    # rcd.recede()
    # print(rcd.slv.getSolutionDict()['x'])
    # for i in range(4):
    #     print('=========== recede ==================')
    #     rcd.recede()
    #

    # !/usr/bin/env python3

    import horizon.problem as prb
    import horizon.utils.plotter as plotter
    import casadi as cs
    import numpy as np
    from horizon.transcriptions.transcriptor import Transcriptor
    from horizon.solvers import solver
    import matplotlib.pyplot as plt

    n_nodes = 25
    dt = 0.1
    mu = 0.2
    grav = 9.81
    prob = prb.Problem(n_nodes, receding=True)

    p = prob.createStateVariable('pos', dim=2)

    ig = np.array([[0.00461538, 0.01807692, 0.03961538, 0.06846154, 0.10384615, 0.145,
                    0.19115385, 0.24153846, 0.29538462, 0.35192308, 0.41038462, 0.47,
                    0.53, 0.58961538, 0.64807692, 0.70461538, 0.75846154, 0.80884615,
                    0.855, 0.89615385, 0.93153846, 0.96038462, 0.98192308, 0.99538462,
                    1., 0.],
                   [0.00461538, 0.01807692, 0.03961538, 0.06846154, 0.10384615, 0.145,
                    0.19115385, 0.24153846, 0.29538462, 0.35192308, 0.41038462, 0.47,
                    0.53, 0.58961538, 0.64807692, 0.70461538, 0.75846154, 0.80884615,
                    0.855, 0.89615385, 0.93153846, 0.96038462, 0.98192308, 0.99538462,
                    1., 0.]])
    p.setInitialGuess(ig)
    v = prob.createStateVariable('vel', dim=2)
    F = prob.createInputVariable('force', dim=2)

    p_tgt = prob.createParameter('pos_goal', dim=2)
    state = prob.getState()
    state_prev = state.getVarOffset(-1)
    x = state.getVars()

    xdot = cs.vertcat(v, F)  # - mu*grav*np.sign(v)
    prob.setDynamics(xdot)
    prob.setDt(dt)

    th = Transcriptor.make_method('multiple_shooting', prob)

    # set initial state (rest in zero)
    p.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)
    v.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)

    # final constraint
    # p.setBounds(lb=[1, 1], ub=[1, 1], nodes=n_nodes)
    prob.createFinalConstraint('goal', p - p_tgt)
    v.setBounds(lb=[0, 0], ub=[0, 0], nodes=n_nodes)

    obs_center = np.array([0.5, 0.5])
    obs_r = 0.4
    obs = cs.sumsqr(p - obs_center) - obs_r ** 2

    # obs_cnsrt = prob.createIntermediateConstraint('obstacle', obs)
    # obs_cnsrt.setUpperBounds(np.inf)

    # intermediate cost ( i want to minimize the force! )
    prob.createIntermediateCost('cost', cs.sumsqr(F), )

    # solve
    traj = numpy.array([])
    solver = solver.Solver.make_solver('ipopt', prob)
    rcd = RecedingHandler(solver)
    p_tgt.assign([1, 1])
    for i in range(30):
        rcd.recede()
        solution = solver.getSolutionDict()
        p_tgt.assign([1, 1])
        prob.getParameters('cost_weight_mask').assign([1])
        # required bounds for setting intial position
        p.setBounds(solution['pos'][:, 1], solution['pos'][:, 1], 0)

        traj = np.hstack((traj, np.atleast_2d(solution['pos'][:, 0]).T)) if traj.size else np.atleast_2d(solution['pos'][:, 0]).T

    print(traj)
    fig, ax = plt.subplots()
    ax.set_title('xy plane')
    ax.plot(traj[0], traj[1])
    ax.plot([0, 0], [0, 0], 'bo', markersize=12)
    ax.plot([1, 1], [1, 1], 'g*', markersize=12)

    plt.show()
    exit()
    # plot
    plot_all = True

    if plot_all:
        hplt = plotter.PlotterHorizon(prob, solution)
        hplt.plotVariables(['pos', 'vel', 'force'], grid=True)
        # hplt.plotFunctions(grid=True)

    fig, ax = plt.subplots()
    ax.set_title('xy plane')
    ax.plot(solution['pos'][0], solution['pos'][1])
    ax.plot([0, 0], [0, 0], 'bo', markersize=12)
    ax.plot([1, 1], [1, 1], 'g*', markersize=12)

    # circle = plt.Circle(obs_center, radius=obs_r, fc='r')
    # ax.add_patch(circle)
    # ax.legend(['traj', 'start', 'goal', 'obstacle'])
    # plt.gca().add_patch(circle)

    plt.show()
