from build_report import h1, h2, body, bullets, code_block


def add(story):
    h1(story, "Marker / Empty Files (one-line each)")
    body(story, "These files exist but contain no executable logic. They are listed "
                "for completeness so the reader knows they were inspected and "
                "intentionally skipped from line-level analysis.")

    bullets(story, [
        "<code>config/__init__.py</code> — empty package marker. Makes "
        "<code>config</code> importable as a module so "
        "<code>from config import config as C</code> resolves.",
        "<code>data/__init__.py</code> — empty package marker for the "
        "<code>data</code> module.",
        "<code>model/__init__.py</code> — empty package marker for the "
        "<code>model</code> module.",
        "<code>rollout/__init__.py</code> — one-line docstring "
        "(<code>\"\"\"Robot-side MicroACT rollout helpers.\"\"\"</code>). No exports.",
        "<code>rollout/adapters/__init__.py</code> — one-line docstring "
        "(<code>\"\"\"Robot adapters for VLA rollout.\"\"\"</code>). No exports.",
        "<code>evaluate.py</code> — empty file (0 bytes). Reserved for a future "
        "standalone evaluator; today validation runs inline in train.py / train_vla.py.",
        "<code>.codex</code> — empty file (0 bytes). Read-only marker used by an "
        "external tool to recognize the directory; not part of MicroACT itself.",
    ])

    h1(story, "dataset_vla/README.md (data layout reference)")
    body(story, "The user requested README files be skipped. This one is included "
                "because it documents the on-disk schema that "
                "<code>data/vla_dataset.py</code> consumes; without seeing the schema "
                "the dataset code is not fully grounded.")
    h2(story, "On-disk layout")
    code_block(story, "dataset_vla/README.md - directory layout", """\
dataset_vla/
└── episodes/
    └── <episode_id>/
        ├── metadata.json
        ├── trajectory.csv
        └── frames/
            └── cam_main/
                ├── frame_000000.png
                └── frame_000001.png""")
    h2(story, "Minimal metadata.json")
    code_block(story, "dataset_vla/README.md - metadata example", """\
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
}""")
    bullets(story, [
        "<b>state_dim / action_dim</b> declare the per-episode shape. The dataset "
        "right-pads to <code>MAX_STATE_DIM = MAX_ACTION_DIM = 16</code> and emits a "
        "boolean mask so single-arm (e.g. dim=4) and dual-arm (dim=8) episodes mix "
        "in the same training batch.",
        "<b>state_cols / action_cols</b> are the CSV column names in the order they "
        "should be packed into <code>qpos[:state_dim]</code> and "
        "<code>action[:, :action_dim]</code>. Missing columns are a hard error.",
        "<b>image_col / timestep_col</b> are the CSV column names that hold the "
        "image path and timestep index. Defaults are <code>\"image_path\"</code> "
        "and <code>\"timestep\"</code>.",
        "<b>frames/&lt;camera_name&gt;/frame_NNNNNN.png</b> is the fallback layout. "
        "If the CSV's <code>image_path</code> column is empty for a row, "
        "<code>_resolve_image_path</code> tries this conventional path before "
        "giving up to a black frame.",
    ])
