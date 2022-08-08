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
prob = prb.Problem(n_nodes, casadi_type=cs.SX)

# dt = 0.1
dt = prob.createInputVariable('dt', dim=1)

x1 = prob.createStateVariable('x1', dim=3)
x2 = prob.createStateVariable('x2', dim=3)


vx = prob.createParameter('vx', 1)
v = prob.createVariable('v', 2)

vx.assign(3)

v_tot = cs.vertcat(vx, v)

state = prob.getState()
state_prev = state.getVarOffset(-1)
x = state.getVars()

xdot = cs.vertcat(v_tot, v_tot) #- mu*grav*np.sign(v)
prob.setDynamics(xdot)
prob.setDt(dt)

th = Transcriptor.make_method('multiple_shooting', prob)

# set initial state (rest in zero)
x1.setBounds([0, 0, 0], [0, 0, 0], nodes=0)
x2.setBounds([0, 5, 0], [0, 5, 0], nodes=0)

# set final state (rest in zero)
# x1.setBounds([10, 0, 0], [10, 0, 0], nodes=n_nodes)
# x2.setBounds([10, 5, 0], [10, 5, 0], nodes=n_nodes)

x1[2].setBounds(0, 0)
x1[2].setBounds(1, 1, nodes=range(2, n_nodes, 4))
x1[2].setBounds(1, 1, nodes=range(3, n_nodes, 4))

v[0].setBounds(0, 0)

if isinstance(dt, horizon.variables.InputVariable):
    dt.setBounds(0.05, 2)
# final constraint

obs_center = np.array([5, 5, 0])
obs_r = 0.3
obs = cs.sumsqr(x2 - obs_center) - obs_r**2
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
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = center[0] + radius * np.cos(u)*np.sin(v)
    y = center[1] + radius * np.sin(u)*np.sin(v)
    z = center[2] + radius * np.cos(v)

    ax.plot_surface(x, y, z, color=np.random.choice(['g','b']), alpha=0.5*np.random.random()+0.5)

hplt = plotter.PlotterHorizon(prob, solution)
hplt.plotVariables(['v'], grid=True)
hplt.plotVariables(['x1'], grid=True)
hplt.plotVariables(['x2'], grid=True)
if isinstance(dt, horizon.variables.InputVariable):
    hplt.plotVariables(['dt'], grid=True)

plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
ax.plot3D(solution['x1'][0], solution['x1'][1], solution['x1'][2])
ax.plot3D(solution['x2'][0], solution['x2'][1], solution['x2'][2])

ax.scatter(solution['x1'][0], solution['x1'][1], solution['x1'][2])
ax.scatter(solution['x2'][0], solution['x2'][1], solution['x2'][2])

plt_sphere(obs_center, obs_r)


plt.show()
