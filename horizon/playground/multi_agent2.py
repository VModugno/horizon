#!/usr/bin/env python3

import horizon.problem as prb
import horizon.utils.plotter as plotter
import casadi as cs
import numpy as np
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from horizon.utils import resampler_trajectory
import math as m
import copy

n_nodes = 4
prob = prb.Problem(n_nodes, casadi_type=cs.SX)

tf = 1.0
# tf = prob.createSingleVariable('tf', dim=1)
dt = tf / n_nodes

# dt = prob.createInputVariable('dt', dim=1)
# tf = dt * n_nodes


alpha1 = prob.createInputVariable('alpha1', dim=1)
alpha2 = prob.createInputVariable('alpha2', dim=1)
# alpha2 = prob.createParameter('alpha2', dim=1)
# alpha2.assign(1.0)

x1 = prob.createStateVariable('x1', dim=1)
x2 = prob.createStateVariable('x2', dim=1)

t1 = prob.createStateVariable('t1', dim=1)
t2 = prob.createStateVariable('t2', dim=1)

v1 = prob.createInputVariable('v1', dim=1)
v2 = prob.createInputVariable('v2', dim=1)

# a1 = prob.createInputVariable('a1', dim=1)
# a2 = prob.createInputVariable('a2', dim=1)

# dt.setBounds(0.0, 10.0)
alpha1.setBounds(0.04, 1000.0)
alpha2.setBounds(0.04, 1000.0)

alpha1.setInitialGuess(0.0)
alpha2.setInitialGuess(1.0)

state = prob.getState()
x = state.getVars()

xdot = cs.vertcat(alpha1*v1, alpha2*v2, alpha1, alpha2) #- mu*grav*np.sign(v)
prob.setDynamics(xdot)
prob.setDt(dt)

u = cs.vertcat(alpha1, alpha2, v1, v2)

th = Transcriptor.make_method('multiple_shooting', prob)

# set initial state (rest in zero)
x1.setBounds(1.0, 1.0, nodes=0)
x1.setBounds(0.0, 0.0, nodes=[1])
x1.setBounds(1.0, 1.0, nodes=[3])

x2.setBounds(1.0, 1.0, nodes=[0])
x2.setBounds(2.0, 2.0, nodes=[2])
x2.setBounds(1.0, 1.0, nodes=4)

v1.setBounds(0.0, 0.0, nodes=[1,3,4]) 
v1.setBounds(-0.5, 0.5, nodes=[2])
v1.setBounds(-np.inf, np.inf, nodes=[0])

v2.setBounds(0.0, 0.0, nodes = [0,2,4])
v2.setBounds(-0.2, 0.2, nodes=[1,3])

# a1.setUpperBounds(1.0, nodes=2)
# a2.setUpperBounds(0.5, nodes=[1,3])

t1.setBounds(0.0,0.0, nodes=0)
t2.setBounds(0.0,0.0, nodes=0)

# prob.createConstraint('ensure_pick_is_finished', alpha1*dt - alpha2*dt, nodes=0)
# prob.createConstraint('ensure_place_is_finished', (alpha1.getVarOffset(-2)*dt+alpha1.getVarOffset(-1)*dt+alpha1*dt) - (alpha2.getVarOffset(-2)*dt+alpha2.getVarOffset(-1)*dt+alpha2*dt), nodes=3)
prob.createFinalConstraint('ensure_place_is_finished', t1-t2)
prob.createConstraint('ensure_pick_is_finished', t1-t2, nodes=1)

prob.createFinalCost('min_tf', t1)
prob.createIntermediateCost('min_u', cs.sumsqr(u[2:4]), nodes=range(1,n_nodes))

# solve
solver = solver.Solver.make_solver('ipopt', prob)
solver.solve()
solution = solver.getSolutionDict()

print("total time x1", np.sum(solution['alpha1'] * dt))
print("total time x2", np.sum(solution['alpha2'] * dt))
t1 = solution['t1'].flatten()
t2 = solution['t2'].flatten()
print("cumulated t1: ", t1)
print("cumulated t2: ", t2)

# plot
plot_all = True

# if plot_all:
hplt = plotter.PlotterHorizon(prob, solution)
hplt.plotVariables(['x1', 'x2', 'v1', 'v2', 'alpha1', 'alpha2'], grid=True, markers=True, show_bounds=False)
hplt.plotFunctions(show_bounds=True)

plt.figure()
plt.plot(solution['t1'], solution['x1'], 'r--o')
plt.plot(solution['t2'], solution['x2'], 'b--o')
plt.grid()

# plt.show()

# resampling
resampling = False
if resampling:
    dt1_before_res = (solution['alpha1'] * dt).flatten()
    dt2_before_res = (solution['alpha2'] * dt).flatten()

    print(dt1_before_res)

    dt_res = 0.001
    
    xdot_scaled = cs.vertcat(v1, v2)
    dae = {'x': x[:2], 'p': u[2:4], 'ode': xdot_scaled[:2], 'quad': 1}

    # v1_scaled = solver.u_opt[2,:]
    # print("v1 real: ", v1_scaled)
    # v2_scaled = solver.u_opt[3,:]
    # print("v2 real: ", v2_scaled)
    # v_scaled = np.vstack((v1_scaled.T, v2_scaled.T))
    # print("v real: ", v_scaled)

    state_res_1 = resampler_trajectory.resampler(solver.x_opt[:2,:], solver.u_opt[2:4,:], dt1_before_res, dt_res, dae)
    state_res_2 = resampler_trajectory.resampler(solver.x_opt[:2,:], solver.u_opt[2:4,:], dt2_before_res, dt_res, dae)

    plt.figure()
    plt.plot((state_res_1)[0, :])
    plt.plot((state_res_2)[1, :])
    plt.show()
        
##### 2nd round
dt_new = 0.01
tf_new = t1[-1].item()
tf_new = m.ceil(tf_new * 100) / 100.0
print("tf_new: ", tf_new)
n_nodes_new = m.ceil(tf_new/dt_new)
print("n_nodes_new: ", n_nodes_new)

prob_new = prob
# prob_new = prb.Problem(n_nodes_new, casadi_type=cs.SX)
prob_new.setNNodes(n_nodes_new)

x_old = prob.getState()
vars = prob.getVariables()
x_to_remove = ('t1', 't2')
for entry in x_to_remove:
    x_old.removeVariable(entry)
    vars.pop(entry)
u_old = prob.getInput()
u_to_remove = ('alpha1', 'alpha2')
for entry in u_to_remove:
    u_old.removeVariable(entry)
    vars.pop(entry)

for k,v in x_old.getVars().items():
    k = prob_new.createStateVariable(k, v.getDim())
for k,v in u_old.getVars().items():
    k = prob_new.createInputVariable(k, v.getDim())

x = prob_new.getState()
u = prob_new.getInput()
    
for node in range(n_nodes+1):
    t = solution['t1'][node]
    print(t)
    k = m.floor(t/dt_new)
    print("significative k: ", k)
    lb, ub = x_old.getBounds(node)
    x.setBounds(lb, ub, nodes=[k])
    lb, ub = u_old.getBounds(node)
    u.setBounds(lb, ub, nodes=[k])

prob_new.createFinalCost('min_u', u)