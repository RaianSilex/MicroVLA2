# MicroVLA Dataset Layout

Place heterogeneous demonstrations under:

```text
dataset_vla/
└── episodes/
    └── <episode_id>/
        ├── metadata.json
        ├── trajectory.csv
        └── frames/
            └── cam_main/
                ├── frame_000000.png
                └── frame_000001.png
```

Minimal `metadata.json`:

```json
{
  "episode_id": "lab_a_trial_0001",
  "lab_id": "lab_a",
  "robot_id": "sensapex_dual_ump4",
  "embodiment": "dual_manipulator",
  "action_type": "absolute_position",
  "task_family": "cell_manipulation",
  "instruction": "move both manipulators toward the selected cell",
  "camera_names": ["cam_main"],
  "state_dim": 8,
  "action_dim": 8,
  "state_cols": [
    "current_x", "current_y", "current_z", "current_d",
    "current_x2", "current_y2", "current_z2", "current_d2"
  ],
  "action_cols": [
    "target_x", "target_y", "target_z", "target_d",
    "target_x2", "target_y2", "target_z2", "target_d2"
  ],
  "image_col": "image_path",
  "timestep_col": "timestep"
}
```

Single-manipulator episodes can use `state_dim: 4` and `action_dim: 4`.
The loader pads all state/action tensors to the shared VLA maximums and masks
invalid dimensions during loss computation.
