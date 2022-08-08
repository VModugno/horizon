#!/usr/bin/env python3

import horizon.problem as prb
import horizon.utils.plotter as plotter
import casadi as cs
import numpy as np

import horizon.variables
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
import matplotlib.pyplot as plt

n_nodes = 20
prob = prb.Problem(n_nodes)

dt1 = prob.createInputVariable('dt1', dim=1)
dt2 = prob.createInputVariable('dt2', dim=1)

x1 = prob.createStateVariable('x1', dim=3)
x2 = prob.createStateVariable('x2', dim=3)

vx1 = prob.createParameter('vx1', 1)
v1 = prob.createInputVariable('v1', 2)

vx2 = prob.createParameter('vx2', 1)
v2 = prob.createInputVariable('v2', 2)

vx1.assign(3)
vx2.assign(3)

v_tot1 = horizon.variables.Aggregate()
v_tot1.addVariable(vx1)
v_tot1.addVariable(v1)
#
v_tot2 = horizon.variables.Aggregate()
v_tot2.addVariable(vx2)
v_tot2.addVariable(v2)

state = prob.getState()
state_prev = state.getVarOffset(-1)
x = state.getVars()

# aggregate loses type
# xdot = cs.vertcat(v_tot1.getVars())
xdot = cs.vertcat(v_tot1.getVars(), v_tot2.getVars())
prob.setDynamics(xdot)

goal_x = 40
# prob.setDt(dt1)
th = Transcriptor.make_method('multiple_shooting', prob)

# add integrators
import horizon.transcriptions.integrators as integrators

dae1 = {'x': x1, 'p': v_tot1.getVars(), 'ode': v_tot1.getVars(), 'quad': 0}
f_int1 = integrators.__dict__['RK4'](dae1, {}, cs.SX)

th.add_ms_to_var(x1, v_tot1, dt1, f_int1)

# set initial state (rest in zero)
x1.setBounds([0., 0., 0.], [0., 0., 0.], nodes=0)

# set final state
x1[0].setLowerBounds(goal_x, nodes=n_nodes)


x1[2].setBounds(0, 0)
x1[2].setBounds(1, 1, nodes=range(2, n_nodes, 4))
x1[2].setBounds(1, 1, nodes=range(3, n_nodes, 4))

v1[0].setBounds(0, 0)
if isinstance(dt1, horizon.variables.InputVariable):
    dt1.setBounds(0.01, 2)
# ================================================
# ================================================
# ================================================
#
dae2 = {'x': x2, 'p': v_tot2.getVars(), 'ode': v_tot2.getVars(), 'quad': 0}
f_int2 = integrators.__dict__['RK4'](dae2, {}, cs.SX)

th.add_ms_to_var(x2, v_tot2, dt2, f_int2)
x2.setBounds([0., 5., 1.], [0., 5., 1.], nodes=0)

# set final state
x2[0].setLowerBounds(goal_x, nodes=n_nodes)


x2[2].setBounds(1, 1)
x2[2].setBounds(0, 0, nodes=range(2, n_nodes, 4))
x2[2].setBounds(0, 0, nodes=range(3, n_nodes, 4))

v2[0].setBounds(0, 0)
if isinstance(dt2, horizon.variables.InputVariable):
    dt2.setBounds(0.05, 2)

# ================================================
# final constraint

obs_center = np.array([15, 5, 0])
obs_r = 0.5
obs = cs.sumsqr(x2 - obs_center) - obs_r ** 2
#
obs_cnsrt = prob.createIntermediateConstraint('obstacle', obs)
obs_cnsrt.setUpperBounds(np.inf)

# solve
solver = solver.Solver.make_solver('ipopt', prob)
solver.solve()
solution = solver.getSolutionDict()

# plot
def plt_sphere(center, radius):
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)

    ax.plot_surface(x, y, z, color=np.random.choice(['g', 'b']), alpha=0.5 * np.random.random() + 0.5)


hplt = plotter.PlotterHorizon(prob, solution)
hplt.plotVariables(['v1'], grid=True)
hplt.plotVariables(['v2'], grid=True)
hplt.plotVariables(['x1'], grid=True)
hplt.plotVariables(['x2'], grid=True)

if isinstance(dt1, horizon.variables.InputVariable):
    hplt.plotVariables(['dt1'], grid=True)

    total_time_1 = np.sum(solution['dt1'])
    print(total_time_1)



if isinstance(dt2, horizon.variables.InputVariable):
    hplt.plotVariables(['dt2'], grid=True)

    total_time_2 = np.sum(solution['dt2'])
    print(total_time_2)


plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
ax.plot3D(solution['x1'][0], solution['x1'][1], solution['x1'][2])
ax.plot3D(solution['x2'][0], solution['x2'][1], solution['x2'][2])

ax.scatter(solution['x1'][0], solution['x1'][1], solution['x1'][2])
ax.scatter(solution['x2'][0], solution['x2'][1], solution['x2'][2])

plt_sphere(obs_center, obs_r)

plt.show()
