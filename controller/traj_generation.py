import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd


class DroneTrajectory:
    def __init__(self, init_state, goal_state, t0, tf, gen=True):
        self.p0 = init_state[0]
        self.v0 = init_state[1]
        self.a0 = init_state[2]
        self.yaw0 = init_state[3]  # extrincis euelr ZYX
        self.pf = goal_state[0]
        self.vf = goal_state[1]
        self.af = goal_state[2]
        self.yawf = goal_state[3]
        self.t0 = t0
        self.tf = tf
        self.Gz = 9.81
        self.e3 = np.array([0, 0, 1]).reshape(3, 1)
        self.traj_data = None

        if gen:
            self.compute_polynomial_coefficients(
                self.p0, self.v0, self.a0, self.pf, self.vf, self.af, t0, tf
            )
            self.compute_yaw_coefficients(self.yaw0, self.yawf, t0, tf)

    def load_traj(self, traj_path, dt, times=1):
        self.traj_data = pd.read_csv(
            traj_path,
            names=[
                "p_x",
                "p_v",
                "p_z",
                "v_x",
                "v_y",
                "v_z",
                "a_x",
                "a_y",
                "a_z",
                "q_x",
                "q_y",
                "q_z",
                "q_w",
                "w_x",
                "w_y",
                "w_z",
                "a0",
                "a1",
                "a2",
                "a3",
            ],
        )
        self.dt = dt
        self.tf = (len(self.traj_data) - 1) * dt * times
        self.traj_times = times

    def deuler2omega(self, angle, dangle):
        """
        omega = A @ dangle
        angle = [roll, pitch, yaw], extrinsic ZYX (rad)
        dangle = [droll, dpitch, dyaw] (rad/s)
        """
        roll, pitch, yaw = angle[0], angle[1], angle[2]
        droll, dpitch, dyaw = dangle[0], dangle[1], dangle[2]
        A = np.array(
            [
                [1, 0, -np.sin(pitch)],
                [0, np.cos(roll), np.sin(roll) * np.cos(pitch)],
                [0, -np.sin(roll), np.cos(roll) * np.cos(pitch)],
            ]
        )
        omega = A @ np.array([droll, dpitch, dyaw]).reshape(3, 1)
        return omega.reshape(3)

    def compute_polynomial_coefficients(self, p0, v0, a0, pf, vf, af, t0, tf):
        T_mat = np.array(
            [
                [1, t0, t0**2, t0**3, t0**4, t0**5],
                [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
                [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
            ]
        )

        coeffs = []
        for i in range(3):
            b = np.array([p0[i], v0[i], a0[i], pf[i], vf[i], af[i]])
            a = np.linalg.solve(T_mat, b)
            coeffs.append(a)
        self.coeffs = np.array(coeffs)

    def compute_yaw_coefficients(self, yaw0, yawf, t0, tf):
        """
        3차 다항식으로 yaw 궤적 생성: yaw(t) = a0 + a1*t + a2*t^2 + a3*t^3
        조건: yaw0, yawf, dyaw0 = 0, dyawf = 0
        """
        A = np.array(
            [
                [1, t0, t0**2, t0**3],
                [0, 1, 2 * t0, 3 * t0**2],
                [1, tf, tf**2, tf**3],
                [0, 1, 2 * tf, 3 * tf**2],
            ]
        )
        b = np.array([yaw0, 0, yawf, 0])
        self.yaw_coeffs = np.linalg.solve(A, b)

    def get_trajectory(self, t, rotation=False):
        if t > self.tf:
            t = self.tf
        if self.traj_data is not None:
            seq = int((t - self.t0) / (self.dt * self.traj_times))
            data = self.traj_data.values[seq]
            p = data[0:3]
            v = data[3:6]
            a = data[6:9]
            quat = data[9:13]
            tmp = R.from_quat(quat, scalar_first=False)
            Rot = tmp.as_matrix()
            yaw = tmp.as_euler("ZYX", degrees=False)[0]
            if t == self.tf:
                v = [0, 0, 0]
                a = [0, 0, 0]
                Rot = np.array(
                    [
                        [np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1],
                    ]
                )

            if rotation:
                return np.array(p), np.array(v), np.array(a), np.array(Rot)

            return np.array(p), np.array(v), np.array(a), np.array(yaw)

        else:
            p, v, a = [], [], []
            for i in range(3):
                c = self.coeffs[i]
                p.append(np.polyval(c[::-1], t))
                v.append(np.polyval(np.polyder(c[::-1], 1), t))
                a.append(np.polyval(np.polyder(c[::-1], 2), t))

            yaw = np.polyval(self.yaw_coeffs[::-1], t)
            # dyaw = np.polyval(np.polyder(self.yaw_coeffs[::-1], 1), t)
            # ddyaw = np.polyval(np.polyder(self.yaw_coeffs[::-1], 2), t)

            if rotation:

                b1_proj = np.array([np.cos(yaw), np.sin(yaw), 0]).reshape(3, 1)
                b3 = np.array(a).reshape(3, 1) + self.e3 * self.Gz
                b3 /= np.linalg.norm(b3)
                b2 = np.cross(b3.reshape(3), b1_proj.reshape(3)).reshape(3, 1)
                b2 /= np.linalg.norm(b2)
                b1 = np.cross(b2.reshape(3), b3.reshape(3)).reshape(3, 1)
                b1 /= np.linalg.norm(b1)
                Rot = np.hstack([b1, b2, b3])

                return np.array(p), np.array(v), np.array(a), np.array(Rot)

            return np.array(p), np.array(v), np.array(a), np.array(yaw)
