from horizon.rhc.tasks.task import Task
import casadi as cs
from horizon.problem import Problem
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot

# todo name is useless
class CartesianTask(Task):
    def __init__(self, frame, cartesian_type=None, *args, **kwargs):

        self.frame = frame

        self.cartesian_type = 'position' if cartesian_type is None else cartesian_type

        super().__init__(*args, **kwargs)

        self.indices = np.array([0, 1, 2]).astype(int) if self.indices is None else np.array(self.indices).astype(int)

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self._initialize()

    # TODO
    # def _fk(self, frame, q, derivative=0):
    #     if ...
    #     fk = cs.Function.deserialize(self.kin_dyn.fk(frame))
    #     ee_p_t = fk(q=q)['ee_pos']
    #     ee_p_r = fk(q=q)['ee_rot']
    #     return ee_p_t, ee_p_r

    def _rot_to_quat(self, R):

        quat = cs.SX(4, 1) # todo is this ok?
        quat[0] = 1 / 2 * np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
        quat[1] = np.sign(R[2, 1] - R[1, 2] * np.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1))
        quat[2] = np.sign(R[0, 2] - R[2, 0] * np.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1))
        quat[3] = np.sign(R[1, 0] - R[0, 1] * np.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1))

        return quat

    def _quat_to_rot(self, quat):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param quat: A 4 element array representing the quaternion (im(quat), re(quat))

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q1 = quat[0]
        q2 = quat[1]
        q3 = quat[2]
        q0 = quat[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        r0 = cs.horzcat(r00, r01, r02)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        r1 = cs.horzcat(r10, r11, r12)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        r2 = cs.horzcat(r20, r21, r22)

        # 3x3 rotation matrix
        rot_matrix = cs.vertcat(r0, r1, r2)

        return rot_matrix


    def _skew(self, vec):
        skew_op = np.zeros([3, 3])

        skew_op[0, 1] = - vec[2]
        skew_op[0, 2] = vec[1]
        skew_op[1, 0] = vec[2]
        skew_op[1, 2] = - vec[0]
        skew_op[2, 0] = - vec[1]
        skew_op[2, 1] = vec[0]

        return skew_op

    def _compute_orientation_error2(self, val1, val2):

        # siciliano's method
        quat_1 = self._rot_to_quat(val1)
        if val2.shape == (3, 3):
            quat_2 = self._rot_to_quat(val2)
        elif val2.shape == (4, 1):
            quat_2 = val2

        rot_err = quat_1[3] * quat_2[0:3] - \
                  quat_2[3] * quat_1[0:3] - \
                  cs.mtimes(self._skew(quat_2[0:3]), quat_1[0:3])

        return rot_err

    def _compute_orientation_error(self, R_0, R_1, epsi=1e-5):

        R_err = cs.mtimes(R_0, R_1.T)
        R_skew = (R_err - R_err.T) / 2

        r = cs.vertcat(R_skew[2, 1], R_skew[0, 2], R_skew[1, 0])

        div = cs.sqrt(epsi + 1 + cs.trace(R_err))
        rot_error = r / div

        return rot_error

    def _initialize(self):
        # todo this is wrong! how to get these variables?
        q = self.prb.getVariables('q')
        v = self.prb.getVariables('v')

        # kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        if self.cartesian_type == 'position':
            fk = cs.Function.deserialize(self.kin_dyn.fk(self.frame))
            ee_p = fk(q=q)
            ee_p_t = ee_p['ee_pos']
            ee_p_r = ee_p['ee_rot']

            frame_name = f'{self.name}_{self.frame}_pos'
            self.pos_tgt = self.prb.createParameter(f'{frame_name}_tgt', 7)

            fun_trans = ee_p_t - self.pos_tgt[:3]

            # todo why with norm_2 is faster?
            # fun_lin = cs.norm_2(self._compute_orientation_error(ee_p_r, self._quat_to_rot(self.pos_tgt[3:])))
            fun_lin = self._compute_orientation_error(ee_p_r, self._quat_to_rot(self.pos_tgt[3:]))

            # todo this is ugly, but for now keep it
            #   find a way to check if also rotation is involved
            if self.indices.size > 3:
                fun = cs.vertcat(fun_trans, fun_lin)
            else:
                fun = fun_trans

        elif self.cartesian_type == 'velocity':
            dfk = cs.Function.deserialize(self.kin_dyn.frameVelocity(self.frame, self.kd_frame))
            ee_v_t = dfk(q=q, qdot=v)['ee_vel_linear']
            ee_v_r = dfk(q=q, qdot=v)['ee_vel_angular']
            ee_v = cs.vertcat(ee_v_t, ee_v_r)

            frame_name = f'{self.name}_{self.frame}_vel'
            self.vel_tgt = self.prb.createParameter(f'{frame_name}_tgt', self.indices.size)
            fun = ee_v[self.indices] - self.vel_tgt

        elif self.cartesian_type == 'acceleration':
            ddfk = cs.Function.deserialize(self.kin_dyn.frameAcceleration(self.frame, self.kd_frame))
            ee_a_t = ddfk(q=q, qdot=v)['ee_acc_linear']
            ee_a_r = ddfk(q=q, qdot=v)['ee_acc_angular']
            ee_a = cs.vertcat(ee_a_t, ee_a_r)
            frame_name = f'{self.name}_{self.frame}_acc'
            self.acc_tgt = self.prb.createParameter(f'{frame_name}_tgt', self.indices.size)
            fun = ee_a[self.indices] - self.acc_tgt

        self.constr = self.instantiator(f'{frame_name}_task', self.weight * fun, nodes=self.nodes)
        # todo should I keep track of the nodes here?
        #  in other words: should be setNodes resetting?

        # todo initialize well ref (right now only position)
        self.ref = self.pos_tgt[self.indices]

    def getConstraint(self):
        return self.constr

    def setRef(self, ref_traj):

        # if self.cartesian_type == 'position':
        #     cnsrt = self.pos_constr
        # elif self.cartesian_type == 'velocity':
        #     cnsrt = self.vel_constr
        # elif self.cartesian_type == 'acceleration':
        #     cnsrt = self.acc_constr

        ref_matrix = np.atleast_2d(ref_traj)

        # if ref_matrix.shape[0] != cnsrt.getDim():
        #     raise ValueError(f'Wrong goal dimension inserted: ({ref_matrix.shape[0]} != {cnsrt.getDim()})')
        # todo: add this in the initialization
        self.ref.assign(ref_matrix.flatten())

    def setNodes(self, nodes):
        super().setNodes(nodes)

        if not nodes:
            self.nodes = []
            self.constr.setNodes(self.nodes, erasing=True)
            return 0

        # todo when to activate manual mode?

        if self.ref is not None and self.ref.shape[1] != len(self.nodes):
            raise ValueError(f'Wrong nodes dimension inserted: ({self.ref.shape[1]} != {len(self.nodes)})')

        if self.cartesian_type == 'position':
            tgt = self.pos_tgt
        elif self.cartesian_type == 'velocity':
            tgt = self.vel_tgt
        elif self.cartesian_type == 'acceleration':
            tgt = self.acc_tgt
        else:
            raise Exception('Wrong cartesian type')

        # core
        if self.ref is not None:
            tgt.assign(self.ref, nodes=self.nodes)  # <==== SET TARGET
        self.constr.setNodes(self.nodes, erasing=True)  # <==== SET NODES

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')


