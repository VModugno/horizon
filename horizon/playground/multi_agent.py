#!/usr/bin/env python3

import horizon.problem as prb
import horizon.utils.plotter as plotter
import casadi as cs
import numpy as np
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_nodes = 10
tf = 10.0
dt = tf / n_nodes
prob = prb.Problem(n_nodes, casadi_type=cs.SX)

# dt = prob.createInputVariable('dt', dim=1)

alpha1 = prob.createInputVariable('alpha1', dim=1)
alpha2 = prob.createInputVariable('alpha2', dim=1)

x1 = prob.createStateVariable('x1', dim=2)
x2 = prob.createStateVariable('x2', dim=2)

v1 = prob.createVariable('v1', 2)
v2 = prob.createVariable('v2', 2)

v1[0].setBounds(1.0, 1.0)
v2[0].setBounds(1.0, 1.0)

# dt.setBounds(0.1, 2)
alpha1.setBounds(0.0, 2.0)
alpha2.setBounds(0.0, 2.0)

state = prob.getState()
state_prev = state.getVarOffset(-1)
x = state.getVars()

xdot = cs.vertcat(alpha1*v1, alpha2*v2) #- mu*grav*np.sign(v)
prob.setDynamics(xdot)
prob.setDt(dt)

th = Transcriptor.make_method('multiple_shooting', prob)

# set initial state (rest in zero)
x1.setBounds([0, 0], [0, 0], nodes=0)
x2.setBounds([0, 0], [0, 0], nodes=0)

# set final state (rest in zero)
x1.setBounds([10, 0], [10, 0], nodes=n_nodes)
x2.setBounds([10, 0], [10, 0], nodes=n_nodes)

x1[1].setBounds(1, 1, nodes=[num for num in range(1, n_nodes) if num % 2 == 0]) #[num for num in range(1, n_nodes) if num % 2 == 0]
x1[1].setBounds(0, 0, nodes=[num for num in range(1, n_nodes) if num % 2 != 0]) #[num for num in range(1, n_nodes) if num % 2 == 0]
x2[1].setBounds(1, 1, nodes=[num for num in range(1, n_nodes) if num % 2 == 0]) #[num for num in range(1, n_nodes) if num % 2 == 0]
x2[1].setBounds(0, 0, nodes=[num for num in range(1, n_nodes) if num % 2 != 0]) #[num for num in range(1, n_nodes) if num % 2 == 0]
# x2[2].setLowerBounds(0.1, nodes=[4, 5, 6]) #[num for num in range(1, n_nodes) if num % 2 == 0]

# final constraint

obs_center = np.array([5, 0])
obs_r = 0.5
obs = cs.sumsqr(x1 - obs_center)
#
obs_cnsrt = prob.createIntermediateConstraint('obstacle', obs)
obs_cnsrt.setLowerBounds(obs_r**2)
obs_cnsrt.setUpperBounds(np.inf)

# prob.createIntermediateResidual('min_u', 1e1 * dt)
prob.createIntermediateResidual('min_v1', cs.sumsqr(v1)+cs.sumsqr(v2))

# solve
solver = solver.Solver.make_solver('ipopt', prob)
solver.solve()
solution = solver.getSolutionDict()

print("total time x1", np.sum(solution['alpha1'] * dt))
print("total time x2", np.sum(solution['alpha2'] * dt))

# plot
plot_all = True

# if plot_all:
hplt = plotter.PlotterHorizon(prob, solution)
hplt.plotVariables(['x1', 'x2', 'alpha1', 'alpha2'], grid=True, markers=True, show_bounds=False)
hplt.plotFunctions(['obstacle'], show_bounds=True)

plt.figure()
ax = plt.axes(projection='3d')


# Data for a three-dimensional line
ax.plot3D(solution['x1'][0], solution['x1'][1])
ax.plot3D(solution['x2'][0], solution['x2'][1])

# ax.plot([0, 0], [0, 0], 'bo', markersize=12)
# ax.plot([1, 1], [1, 1], 'g*', markersize=12)
# circle = plt.Circle(obs_center, radius=obs_r, fc='r')
# ax.add_patch(circle)
# ax.legend(['traj', 'start', 'goal', 'obstacle'])
# plt.gca().add_patch(circle)

plt.show()
