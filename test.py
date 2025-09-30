import numpy as np
from DroneHoverPolicy import DroneHoverPolicy

if __name__ == "__main__":
    policy = DroneHoverPolicy("/home/mlic/Repo/OmniDrone/policy.pt")

    # fake single-sample inputs
    rel_pos = np.random.randn(3)
    quat = np.random.randn(4)
    vel = np.random.randn(3)
    ang_vel = np.random.randn(3)
    prev_action = np.random.rand(4)

    action = policy.forward(rel_pos, quat, vel, ang_vel, prev_action)
    print("Action:", action)
    print("Action shape:", action.shape)
