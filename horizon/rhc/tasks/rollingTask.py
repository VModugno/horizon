from horizon.rhc.tasks.task import Task
import casadi as cs
import numpy as np


class RollingTask(Task):
    def __init__(self, frame, radius, *args, **kwargs):

        self.frame = frame
        self.radius = radius

        super().__init__(*args, **kwargs)

        self.indices = np.array([0, 1, 2]).astype(
            int) if self.indices is None else np.array(self.indices).astype(int)

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self._initialize()


    def _initialize(self):
        q = self.prb.getVariables('q')
        v = self.prb.getVariables('v')

        dfk_distal = self.kin_dyn.frameVelocity(self.frame, self.kd_frame)
        ee_v_distal_t = dfk_distal(q=q, qdot=v)['ee_vel_linear']
        ee_v_distal_r = dfk_distal(q=q, qdot=v)['ee_vel_angular']

        world_contact_plane_normal = np.zeros(3)
        world_contact_plane_normal[2] = 1.

        wheel_axis = np.zeros(3)
        wheel_axis[2] = 1.

        omega = ee_v_distal_r
        r = - world_contact_plane_normal * self.radius

        v_contact_point = ee_v_distal_t + cs.cross(omega, r)

        self.constr = self.instantiator(f'{self.name}_rolling_task', self.weight * v_contact_point, nodes=self.nodes)


    def setNodes(self, nodes):
        self.constr.setNodes(nodes)

