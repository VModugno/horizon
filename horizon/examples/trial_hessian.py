import casadi as cs
from horizon.utils import utils, kin_dyn, mat_storer

def RK4(dae, opts=None, casadi_type=cs.SX):

  if opts is None:
    opts = dict()
  x = dae['x']
  qddot = dae['p']
  xdot = dae['ode']
  L = dae['quad']

  f_RK = cs.Function('f_RK', [x, qddot], [xdot, L])

  nx = x.size1()
  nv = qddot.size1()

  X0_RK = casadi_type.sym('X0_RK', nx)
  U_RK = casadi_type.sym('U_RK', nv)

  if 'tf' in opts:
    DT_RK = opts['tf']
  else:
    DT_RK = casadi_type.sym('DT_RK', 1)

  X_RK = X0_RK
  Q_RK = 0

  k1, k1_q = f_RK(X_RK, U_RK)
  k2, k2_q = f_RK(X_RK + DT_RK / 2. * k1, U_RK)
  k3, k3_q = f_RK(X_RK + DT_RK / 2. * k2, U_RK)
  k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK)

  X_RK = X_RK + DT_RK / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
  Q_RK = Q_RK + DT_RK / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)

  if 'tf' in opts:
    f = cs.Function('F_RK', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])
  else:
    f = cs.Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])
  return f

q = cs.SX.sym("q", 1)
qdot = cs.SX.sym("qdot", 1)
qddot = cs.SX.sym("qddot", 1)

# Creates double integrator
x, xdot = utils.double_integrator(q, qdot, qddot)

dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': 1}

f = RK4(dae, opts=dict(tf=0.01))
N = 4

for i in range(N):
  q_impl = q = cs.SX.sym(f"q_{i}", 1)
  qdot_impl = q = cs.SX.sym(f"q_{i}", 1)
  qddot_impl = q = cs.SX.sym(f"q_{i}", 1)


F = f.map(N)
print(F)

F = f.map(N,"thread",2)
print(F)

# Xn = F(X, )
# ======================================
# X = cs.vertcat(my_variables)
# u_data = cs.vertcat( .... )
#
# res = my_fun.map(N, 'thread', 4)
# Xn = res(X, u_data), #params);
#
# #multiple shooting
# gaps = Xn[:,1:-1] - X[:,2:]
#
# # V = cs.veccat(params, X); # or only X
#
# nlp = dict('x',X, 'f',1,'g',cs.vec(gaps))

# class MyCallback(cs.Callback):
#     j = None
#     def __init__(self, name, fun, opts={}):
#         cs.Callback.__init__(self)
#         self.fun = fun
#         self.construct(name, opts)
#         if MyCallback.j is None:
#             MyCallback.j = self.fun.jacobian()
#
#     # Number of inputs and outputs
#     def get_n_in(self):
#         return 1
#
#     def get_n_out(self):
#         return 1
#
#     # Initialize the object
#     def init(self):
#         print('initializing object')
#
#     # Evaluate numerically
#     def eval(self, arg):
#         x = arg[0]
#         f = self.fun(x)
#         return [f]
#
#     def has_jacobian(self, *args):
#         return True
#     #
#     def get_jacobian(self, *args):
#         print('poenis')
#         return self.j
#
# if __name__ == '__main__':
#     a = cs.MX.sym('a')
#     my_fun = cs.Function('myFun', [a], [a**2 - 1])
#
#     f = MyCallback('f', my_fun)
#
#     x = cs.MX.sym('x')
#     y = cs.MX.sym('y')
#     j = 1
#     w = a
#     g1 = f(x)
#     g2 = f(y)
#
#     prob_dict = {'f': g1*g2*g1*g2, 'x': cs.vertcat(x, y), 'g': cs.vertcat(g1, g2)}
#
#     # print(prob_dict['g'])
#     # exit()
#     opts = dict()
#     # create solver from prob
#     solver = cs.nlpsol('solver', 'ipopt', prob_dict, opts)
#
#     sol = solver(x0=10)
#
#     # jac.call(2)
#
#
# # x = cs.MX.sym("x")
# # y = cs.MX.sym("y")
# # z = x + y
# # objective = cs.Function('a', [x], [z])
# # #solve NLP and get the expected result
# #
# # jacobian = objective.jacobian()
# # hessian = objective.hessian()
