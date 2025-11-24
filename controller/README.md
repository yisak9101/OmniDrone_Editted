# Controller & Trajectory generation

## Trajectory genretation

### 1. Define
```
traj = DroneTrajectory(start_state, goal_state, t0, tf)
```
- ``start_state`` : initial state of drone (10-dim)
  - position (3-dim) [m]
  - linear velocity (3-dim) [m/s]
  - linear acceleration (3-dim) [m/s^2]
  - yaw (1-dim) [rad]
- ``goal_state`` : goal state of drone (same dimension with ``start_state``)
- ``t0`` : initial timestamp [sec]
- ``tf`` : goal timestamp [sec]

### 2. Trajectory generation
```
p_d, v_d, a_d, yaw_d = traj.get_trajectory(t_current, rotation=False)
# (optional)
_, _, _, rot_d = traj.get_trajectory(t, rotation=True)
```
- ``get_trajectory()`` : compute trajectory of drone at ``t_current``([sec]) timestamp
  - ``rotation`` : `True` -> return desired ratoation, `False` -> return desired yaw

- return variables
  - ``p_d`` : desried position (reference position) at ``t_current`` timestamp [m]
  - ``v_d`` : desried linear velocity (reference linear velocity) at ``t_current`` timestamp [m/s]
  - ``a_d`` : desried linear acceleration (reference linear acceleration) at ``t_current`` timestamp [m/s^2]
  - ``yaw_d`` : desried yaw (reference yaw) at ``t_current`` timestamp [rad]
  - ``rot_d`` : desried rotation (reference rotation) at ``t_current`` timestamp
</br>

## Lee Controller

### 1. Define
```
ctrl = LeeController(mass, f_max, f_min, omega_max)
```
- ``mass``: mass of drone [kg]
- ``f_max`` : minimun collective thrust [N]
- ``f_min`` : maximum collective thrust [N]
- ``omega_max`` : absolute maximun range of body angular velocity command [rad/s]

### 2. Usage
```
cmd, _ = ctrl.compute_control(cur_state, des_state, type="norm_input")
```
- ``cur_state`` : current state of drone
  - position (3-dim) [m]
  - linear velocity (3-dim) [m/s]
  - vectorized rotation matrix : vec(R)
    - vec(R) = ```np.hstack([rot[:, 0], rot[:, 1], rot[:, 2]])```
    - ``rot`` is rotation matrix (3 X 3)
- ``des_state`` : desired state of drone
  - position (3-dim) [m]
  - linear velocity (3-dim) [m/s]
  - linear acceleration (3-dim) [m/s^2]
  - yaw (1-dim) [rad]
- ``type`` : return type (use `norm_input`)
  - ``norm_input`` : scale of ``cmd`` is changed between [-1, 1]
  - ``real_input`` : real value of ``cmd`` is returned
- ``cmd`` : control command [f, omega] (4-dim)
  - collective thrsut command [N] (1-dim)
  - body angular velocity command [rad/s] (3-dim)
### 3. Weight tuning
```
ctrl.KP = np.diag([1, 1, 2]) * 0.1  # 69.44
ctrl.KV = np.diag([1, 1, 2]) * 0.1  # 24.304
ctrl.KR = np.diag([2, 2, 1]) * 5  # 8.81
```
- self.KP : weight for position error
- self.KV : weight for linear velocirt error
- self.KR : weight for rotation error


## Flight test

### 1. Calculate [f, omega] command using trajectory and Lee controller
```
# state feedback
p = drone_position
v = drone_linear velocity
rot = drone rotation matrix
vec_rot = np.hstack([rot[:, 0], rot[:, 1], rot[:, 2]])
cur_state = [p, v, vec_rot]

# trajectory generation
p_d, v_d, a_d, yaw_d = traj.get_trajectory(t_current, rotation=False)
des_state = [p_d, v_d, a_d, yaw_d]

# compute controller
cmd, _ = ctrl.compute_control(cur_state, des_state, type="norm_input")
```
- calculate [f, omgea] command to tracking the reference trajectory (p_d, v_d, a_d, yaw_d)

### 2. Transfer [f, omega] command to rate_controller in simulation
```
# compute controller
cmd, _ = ctrl.compute_control(cur_state, des_state, type="norm_input")

# transfer command to rate_controller
thrusts = rate_controller(cmd, [current state for rate_controller])
```
- calculate [thrusts - inidividual thrusts] command to tracking the [f, omgea] command


### 3. Verify the controller
- check the drone flight
- analyze response properties
  1. rate controller
  - compare [f, omgea] of drone with [f, omgea] command input of rate controller
  - change the weight rate controller
  </br>
  </br>
  2. Lee controller
  - check that drone tracks the reference trajectory (p_d, v_d, a_d, yaw_d)
  - change the weight Lee controller
