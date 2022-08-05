#!/usr/bin/env python3

import horizon.problem as prb
import horizon.utils.plotter as plotter
import casadi as cs
import numpy as np
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
import matplotlib.pyplot as plt

n_nodes = 10
prob = prb.Problem(n_nodes, casadi_type=cs.SX)

dt = prob.createInputVariable('dt', dim=1)

x1 = prob.createStateVariable('x1', dim=3)
x2 = prob.createStateVariable('x2', dim=3)


v = prob.createVariable('v', 3)


state = prob.getState()
state_prev = state.getVarOffset(-1)
x = state.getVars()

xdot = cs.vertcat(v, v) #- mu*grav*np.sign(v)
prob.setDynamics(xdot)
prob.setDt(dt)

th = Transcriptor.make_method('multiple_shooting', prob)

# set initial state (rest in zero)
x1.setBounds([0, 0, 0], [0, 0, 0], nodes=0)
x2.setBounds([0, 5, 0], [0, 5, 0], nodes=0)

# set final state (rest in zero)
x1.setBounds([10, 0, 0], [10, 0, 0], nodes=n_nodes)
x2.setBounds([10, 5, 0], [10, 5, 0], nodes=n_nodes)

x1[2].setBounds(1, 1, nodes=[num for num in range(1, n_nodes) if num % 2 == 0]) #[num for num in range(1, n_nodes) if num % 2 == 0]
x1[2].setBounds(0, 0, nodes=[num for num in range(1, n_nodes) if num % 2 != 0]) #[num for num in range(1, n_nodes) if num % 2 == 0]
# x2[2].setLowerBounds(0.1, nodes=[4, 5, 6]) #[num for num in range(1, n_nodes) if num % 2 == 0]

v[1].setBounds(0, 0)

dt.setBounds(0.1, 2)
# final constraint

obs_center = np.array([0, 5, 0])
obs_r = 0.2
obs = cs.sumsqr(x1 - obs_center) - obs_r**2
#
obs_cnsrt = prob.createIntermediateConstraint('obstacle', obs)
obs_cnsrt.setUpperBounds(np.inf)



# solve
solver = solver.Solver.make_solver('ipopt', prob)
solver.solve()
solution = solver.getSolutionDict()

# plot
plot_all = True

# if plot_all:
hplt = plotter.PlotterHorizon(prob, solution)
hplt.plotVariables(['v'], grid=True)

plt.figure()
ax = plt.axes(projection='3d')


# Data for a three-dimensional line
ax.plot3D(solution['x1'][0], solution['x1'][1], solution['x1'][2])
ax.plot3D(solution['x2'][0], solution['x2'][1], solution['x2'][2])

# ax.plot([0, 0], [0, 0], 'bo', markersize=12)
# ax.plot([1, 1], [1, 1], 'g*', markersize=12)
# circle = plt.Circle(obs_center, radius=obs_r, fc='r')
# ax.add_patch(circle)
# ax.legend(['traj', 'start', 'goal', 'obstacle'])
# plt.gca().add_patch(circle)

plt.show()
