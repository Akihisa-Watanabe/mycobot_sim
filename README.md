## Overview
This document explains motion data of this robot movement shown in the video below. The data is stored in `robot_trajectory.npz` file. The robot has 6 joints that move in different ways.

https://github.com/user-attachments/assets/66353505-3ed9-4e9e-90b2-fb54ff5d01c5

## Joint Configuration
When you load the trajectory data using `np.load('robot_trajectory.npz')`, the `joint_angles` array shows how each joint moves over time.

Each joint corresponds to a specific index in the data:
```python
joint_angles[:, 0]  # Base rotation joint
joint_angles[:, 1]  # First arm joint
joint_angles[:, 2]  # Second arm joint
joint_angles[:, 3]  # Third arm joint
joint_angles[:, 4]  # Wrist pitch joint
joint_angles[:, 5]  # Wrist roll joint
```

## Basic Usage Example
Here's how to load and check the joint angles:

```python
import numpy as np

# Load the data
data = np.load('robot_trajectory.npz')

# Get joint angles
joint_angles = data['joint_angles']

# Print base rotation angle at first time step
print(f"Base rotation: {joint_angles[0, 0]} radians")

# Print wrist roll angle at first time step
print(f"Wrist roll: {joint_angles[0, 5]} radians")
```

## Data Structure
The `robot_trajectory.npz` file contains:
- `joint_angles`: Joint positions over time
- `time`: Time stamps
- `end_effector_pos`: Robot hand position
- `error`: Distance from target
- `target_position`: Goal position
