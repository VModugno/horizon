from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.transcriptions.integrators import RK4
import casadi as cs
import numpy as np

N = 5
dt = 0.1
prob = Problem(N, casadi_type=cs.MX)
prob.setDt(dt)

# a random linear dynamics
x1 = prob.createStateVariable('x1', dim=2)
x2 = prob.createStateVariable('x2', dim=2)
u1 = prob.createInputVariable('u1', dim=2)
u2 = prob.createInputVariable('u2', dim=2)

print(type(x1))
print(type(x2))
print(type(u1))
print(type(u2))
exit()
x = prob.getState().getVars()
xdot = cs.vertcat(x2, u1 )#+ 10e-10 * u2)

prob.setDynamics(xdot)
# a random cost
prob.createIntermediateCost('c12', cs.sumsqr(x1 + u2))

# a final constraint
xtgt = np.array([1, 2, 3, 4])
prob.createFinalConstraint('xtgt', x - xtgt)

# an initial state
x0 = -xtgt
prob.getState().setBounds(lb=x0, ub=x0, nodes=0)
prob.getState().setBounds(lb=xtgt, ub=xtgt, nodes=N)
prob.getState().setInitialGuess(x0)



state_list = prob.getState()
state_prev_list = list()

for var in state_list:
    state_prev_list.append(var.getVarOffset(-1))

state = cs.vertcat(*state_list)
state_prev = cs.vertcat(*state_prev_list)

input_list = prob.getInput()
input_prev_list = list()
for var in input_list:
    input_prev_list.append(var.getVarOffset(-1))

input = cs.vertcat(*input_list)
input_prev = cs.vertcat(*input_prev_list)

opts = dict()
dae = dict()


dae['x'] = state
dae['p'] = input #u1 #input #u1 #input
dae['ode'] = xdot
dae['quad'] = 0  # note: we don't use the quadrature fn here

integrator = RK4(dae)
# input_prev
state_int = integrator(state_prev, input_prev, dt)[0] # u1.getVarOffset(-1)

ms = prob.createConstraint('multiple_shooting', state_int - state, nodes=range(1, prob.getNNodes()))
#
# th = Transcriptor.make_method('multiple_shooting', prob, opts=dict(integrator='RK4'))


ipopt = Solver.make_solver('ipopt', prob)

ipopt.solve()
xerr = ipopt.x_opt - ipopt.x_opt


# iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
#    0  2.5000000e+01 8.00e+00 4.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
#    1  3.2989942e-04 2.33e-07 9.62e-01  -1.0 9.62e+01  -2.0 1.00e+00 1.00e+00f  1
#    2  5.7115754e-08 1.56e-07 2.62e-04  -1.7 9.82e-03  -1.6 1.00e+00 1.00e+00h  1
#    3  6.7274345e-11 1.47e-13 8.99e-06  -5.7 1.26e-04  -1.1 1.00e+00 1.00e+00h  1
#    4  9.2294878e-15 4.66e-15 1.05e-07  -8.6 4.44e-06  -1.6 1.00e+00 1.00e+00h  1
#    5  1.4291315e-19 2.66e-15 4.14e-10  -8.6 5.24e-08  -2.1 1.00e+00 1.00e+00h  1