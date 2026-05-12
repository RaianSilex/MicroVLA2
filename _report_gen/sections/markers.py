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

    h1(story, "README.md (pipeline reference)")
    body(story, "The top-level README is not executable, but it is the repo's "
                "operator-facing contract. It confirms the intended Cellpose4 "
                "usage, the supported backbones, the training/rollout commands, "
                "and the MicroVLA extension points that the code sections above "
                "implement.")
    h2(story, "Cellpose4-facing commands and defaults")
    code_block(story, "README.md - Cellpose4 examples", """\
python train.py --backbone dinov2_vits14+cellpose4
python train_vla.py --backbone dinov2_vits14+cellpose4
python viz_summary.py --backbone dinov2_vits14+cellpose4 > arch_dual.txt
python export_onnx.py --backbone dinov2_vits14+cellpose4

CELLPOSE4_DIAMETER = 180.0
CELLPOSE4_CELLPROB_THRESHOLD = -2.0
CELLPOSE4_FLOW_THRESHOLD = 1.5""")
    bullets(story, [
        "<b>Default VLA backbone</b> is documented as "
        "<code>dinov2_vits14+cellpose4</code>: DINOv2 supplies general visual tokens, "
        "while Cellpose-SAM supplies compact cell-aware tokens.",
        "<b>Token budget</b> matches the code: DINOv2 ViT-S gives about "
        "<code>17x22 = 374</code> image tokens at the default resolution, and "
        "Cellpose4 diameter scaling gives about <code>5x7 = 35</code> aux tokens, "
        "for roughly <code>409</code> total visual tokens.",
        "<b>Install note</b>: <code>cellpose&gt;=4.0</code> is required only when "
        "using <code>cellpose</code> or <code>cellpose4</code> backbones; the ResNet "
        "baseline can train without it.",
        "<b>README/code alignment</b>: the README says the training path uses CP-SAM "
        "features/readout tensors directly rather than generating masks per batch; "
        "that is exactly what <code>Cellpose4Backbone._forward_neck</code> does.",
    ])

    h1(story, "dataset_vla/README.md (data layout reference)")
    body(story, "This README documents the on-disk schema that "
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

    h1(story, "_report_gen/ (PDF generator source)")
    body(story, "These files generate this PDF. They are not part of the training or "
                "rollout runtime, but they are included here for repo completeness. "
                "The design is deliberately simple: <code>build_report.py</code> owns "
                "ReportLab styles, page footer, helper functions, and section ordering; "
                "each file under <code>_report_gen/sections/</code> appends one topic "
                "to the shared story list.")
    h2(story, "build_report.py")
    code_block(story, "_report_gen/build_report.py - assembly pattern", """\
from sections import preamble, top_level, config_files, data_files
from sections import model_act, model_vla, finetune, rollout_files, viz_files, markers

def main():
    story = []
    title_block(story, "MicroACT + MicroVLA Full Code Explanation Report", subtitle)
    preamble.add(story)
    top_level.add(story)
    config_files.add(story)
    data_files.add(story)
    model_act.add(story)
    model_vla.add(story)
    finetune.add(story)
    rollout_files.add(story)
    viz_files.add(story)
    markers.add(story)
    out = Path(__file__).resolve().parents[1] / "microact_full_code_report_cellpose4.pdf"
    build(out, story)""")
    bullets(story, [
        "<b>Styles</b>: title, subtitle, heading, body, bullet, code-label, and code "
        "paragraph styles are defined once in <code>build_report.py</code>. Section "
        "files call helper functions (<code>h1</code>, <code>body</code>, "
        "<code>bullets</code>, <code>code_block</code>) instead of touching ReportLab "
        "directly.",
        "<b>Output path</b>: the regenerated Cellpose4 report is written into the "
        "current repo root as <code>microact_full_code_report_cellpose4.pdf</code>, "
        "avoiding the old hard-coded <code>/home/raianlaptop/MicroACT</code> path.",
        "<b>Footer</b>: every page gets the same report title and page number through "
        "a ReportLab <code>PageTemplate</code> callback.",
    ])
    h2(story, "Section modules")
    bullets(story, [
        "<code>_report_gen/sections/preamble.py</code> — shared shape legend, "
        "Cellpose4 update coverage, and end-to-end ACT/VLA tensor flow.",
        "<code>_report_gen/sections/top_level.py</code> — requirements, .gitignore, "
        "utils, train.py, train_vla.py, export_onnx.py, and evaluate.py.",
        "<code>_report_gen/sections/config_files.py</code> — ACT config plus VLA config.",
        "<code>_report_gen/sections/data_files.py</code> — homogeneous ACT dataset and "
        "heterogeneous VLA dataset.",
        "<code>_report_gen/sections/model_act.py</code> — transformer primitives, "
        "all image backbones including Cellpose4, ACTCVAE, and ACTPolicy.",
        "<code>_report_gen/sections/model_vla.py</code> — language encoder, embodiment "
        "tokens, VLACVAE, and VLAPolicy.",
        "<code>_report_gen/sections/finetune.py</code> — VLA finetune helpers, freeze "
        "modes, partial state-dict loading, and LoRA.",
        "<code>_report_gen/sections/rollout_files.py</code> — shared rollout helpers, "
        "MicroACT rollout, MicroVLA rollout, ROS/Sensapex client, and adapter.",
        "<code>_report_gen/sections/viz_files.py</code> — torchinfo and torchviz "
        "visualization helpers.",
        "<code>_report_gen/sections/markers.py</code> — marker files, READMEs, and "
        "this generator-source inventory.",
        "<code>_report_gen/sections/__init__.py</code> — empty package marker so "
        "section modules import cleanly.",
    ])
