# class PatternGenerator:
#     def __init__(self, pattern, duty_cycle, t_init, stride_time, step_n_min=3):
#
#         # t_start = t_init
#         # t_goal = t_start + stride_time * (1 - duty_cycle)
#         # s = Step(c, k_start, k_end)
#         # steps.append(s)

steps = list()
n_steps = 8
pattern = [0, 3, 1, 2]
stride_time = 6.0
duty_cycle = 0.0
tinit = 0.
dt = 0.1
nc = 4
for k in range(2):
    for i in range(n_steps):
        l = pattern[i % nc]
        t_start = tinit + i * stride_time / nc
        t_goal = t_start + stride_time * (1 - duty_cycle)
        s = dict(leg=l, k_start=k + int(t_start / dt), k_goal=k + int(t_goal / dt))
        steps.append(s)

for step in steps:
    print(step)