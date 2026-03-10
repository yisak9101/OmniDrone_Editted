import numpy as np
import ctypes
import time
from scipy.spatial.transform import Rotation as R

G_PWM_MAX = 60000
G_PWM_MIN = 1000
PWM2THTRUST_A = 0.091492681
PWM2THTRUST_B = 0.067673604
class LeeController:
    def __init__(self, params):
        mass = params['mass']
        thrust_map = [params['rotor_configuration']['force_constants'][0], 0, 0]
        motor_omega_max = params['rotor_configuration']['max_rotation_velocities'][0]
        motor_omega_min = 0
        f_max = (
            thrust_map[0] * motor_omega_max**2
            + thrust_map[1] * motor_omega_max
            + thrust_map[2]
        ) * 4
        f_min = (
            thrust_map[0] * motor_omega_min**2
            + thrust_map[1] * motor_omega_min
            + thrust_map[2]
        ) * 4
        omega_max = [3.141592, 3.141592, 3.141592]

        self.KI = np.diag([1, 1, 1]) * 0.01 # 69.44
        self.KP = np.diag([1, 1, 2]) * 0.1 # 69.44
        self.KV = np.diag([1, 1, 2]) * 0.1 # 24.304
        self.KR = np.diag([2, 2, 1]) * 5 # 8.81
        self.KW = np.diag([1, 1, 1]) * 1 # 2.54
        self.Gz = 9.81
        self.mass = mass
        self.mg = self.mass * self.Gz
        self.f_max = f_max
        self.f_min = f_min
        self.omega_max = np.array(omega_max).reshape(3)
        self.f_std = (f_max - f_min) / 2
        self.f_mean = (f_max + f_min) / 2
        self.e3 = np.array([0, 0, 1]).reshape(3,1)

    def vee_map(self, so3):
        r3 = np.array([-so3[1,2], so3[0,2], -so3[0,1]]).reshape(3,1)
        return r3

    def SO3_error(self, rot, rot_d):
        tmp = rot_d.T @ rot - rot.T @ rot_d
        vee = self.vee_map(tmp)
        e_R = 0.5 * vee

        return e_R

    def compute_control(self, current_state, desired_state, type='real_input'):
        # feedback
        # current state : p, v, vec(R)[col1, col2. col3]
        p = current_state[0].reshape(3,1)
        v = current_state[1].reshape(3,1)
        vec_rot = current_state[2]
        Rot = np.hstack([vec_rot[0:3].reshape(3,1), vec_rot[3:6].reshape(3,1), vec_rot[6:9].reshape(3,1)])
        # omega = current_state[3].reshape(3,1)

        # desired
        # desired_state : p, v, a, yaw
        p_d = desired_state[0].reshape(3,1)
        v_d = desired_state[1].reshape(3,1)
        a_d = desired_state[2].reshape(3,1)
        yaw_d = desired_state[3]

        # error
        e_p = p -p_d
        e_v = v -v_d
        tmp =  -self.KP @ e_p - self.KV @ e_v + self.mg * self.e3 + self.mass * a_d
        if np.linalg.norm(tmp) <= 1e-6:
            print('b3d error')
        b3_d = tmp / np.linalg.norm(tmp)
        b1_d_proj = np.array([np.cos(yaw_d), np.sin(yaw_d), 0])
        b2_d = np.cross(b3_d.reshape(3), b1_d_proj.reshape(3))
        b2_d /= np.linalg.norm(b2_d)
        b1_d = np.cross(b2_d.reshape(3), b3_d.reshape(3))
        b1_d /= np.linalg.norm(b1_d)
        Rot_d = np.hstack([b1_d.reshape(3,1), b2_d.reshape(3,1), b3_d.reshape(3,1)])
        euler_d = R.from_matrix(Rot_d).as_euler('ZYX', degrees=True)
        euler_d = [euler_d[2], euler_d[1], euler_d[0]]

        e_R = self.SO3_error(Rot, Rot_d)

        # collective thrust
        # f, omega : N, rad/s
        cmd_f = np.max([np.dot(tmp.reshape(3), (Rot @ self.e3).reshape(3)), 0])
        cmd_omega = - self.KR @ e_R
        cmd_omega = cmd_omega.reshape(3)

        if type=='real_input':
            ctrl_input = np.array([cmd_f, cmd_omega[0], cmd_omega[1], cmd_omega[2]], dtype=np.float32).reshape(1,4)
            if abs(cmd_omega[0])==self.omega_max[0]:
                print(f'omega x max')
            if abs(cmd_omega[1])==self.omega_max[1]:
                print(f'omega y max')
            if abs(cmd_omega[2])==self.omega_max[2]:
                print(f'omega z max')
            if cmd_f == 0:
                print(f'cmd f 0 error')
            return ctrl_input, euler_d
        elif type == 'norm_input':
            norm_f = (cmd_f - self.f_mean) / self.f_std
            norm_f = min(1, max(norm_f, -1))
            norm_omega = np.min([np.max([cmd_omega, -self.omega_max], axis=0), self.omega_max], axis=0)
            norm_omega = cmd_omega/ self.omega_max
            ctrl_input = np.array([norm_f, norm_omega[0], norm_omega[1], norm_omega[2]], dtype=np.float32).reshape(1,4)
            return ctrl_input, euler_d
        elif type=='setpoint':
            # setpoint
            pwm_ratio = (-PWM2THTRUST_B + np.sqrt(PWM2THTRUST_B**2 + 4 * PWM2THTRUST_A * cmd_f/4)) / (2 * PWM2THTRUST_A)
            pwm = max(min(pwm_ratio, 1.0), 0.0) * 65535

            # rad/s-> [deg/s]
            cmd_omega = cmd_omega * (180 / np.pi)
            cmd_omega[0] = np.min([np.max([cmd_omega[0], -720]), 720])
            cmd_omega[1] = np.min([np.max([cmd_omega[1], -720]), 720])
            cmd_omega[2] = np.min([np.max([cmd_omega[2], -720]), 720])

            setpoint = [cmd_omega[0], cmd_omega[1], cmd_omega[2], pwm]

            return setpoint
        else:
            print('invalid type')
            return None