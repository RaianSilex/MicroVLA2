# MicroVLA dataset conversion

MicroVLA trains from a **LeRobot-format dataset** (SmolVLA / OpenPI / π0 style).
These scripts convert the raw micromanipulation logs into that format.

Raw layout consumed:

```
dataset/
├── logs/trial_N.csv
├── saved_frames/trial_N/frame_NNNNNN.png
└── instruction_labels.csv        # trial_id, region, [instruction]
```

## v3.0 (MicroVLA / SmolVLA) — the main path

```bash
# Build locally under HF_LEROBOT_HOME (default repo id RaianSilex/microvla_ump_dataset):
python dataset_vla/convert_microact_to_lerobot.py
python dataset_vla/convert_microact_to_lerobot.py --limit-trials 3   # quick subset
python dataset_vla/convert_microact_to_lerobot.py --push-to-hub      # upload (needs HF login)
```

Produces standard LeRobot features so any LeRobot/SmolVLA tooling reads it:

- `observation.images.cam_main` — frame (letterboxed to 540×720 by default)
- `observation.state` — absolute Sensapex state (4-D `[x,y,z,d]` for one
  manipulator by default; 8-D for two — set `NUM_MANIPULATORS` or pass
  `--manipulators 2`)
- `action` — **absolute** Sensapex target, same dims as the state (robot-native)
- `task` — the per-trial instruction
- `observation.resistance` — **only if** the logs carry real `resistance_mohm`
  values (auto-detected; `--no-resistance` to skip)

### Grounded, varied instructions

The `task` string is built from the **target cell's region in the frame**, read
from `dataset/instruction_labels.csv`. The converter scaffolds that file (every
trial defaulting to `center`) on the first run and warns — **edit the `region`
column** per trial (or write a free-text `instruction` to override), then re-run.
Canonical regions: the 3×3 grid `top_left, top, top_right, left, center, right,
bottom_left, bottom, bottom_right`; many aliases normalize automatically
(`middle_left`→`left`, `top_center`→`top`, `lower-right`, `upper-left`, `centre`, …).
Wording varies deterministically per trial so language carries real signal.

### Action space: absolute on disk, delta at train time

Actions are stored **absolute** (matching the robot/ROS commands). Training
converts them to **deltas** relative to the current state
(`train_vla.py --action-space delta`, the default); `VLAPolicy.inference` converts
the predicted delta back to absolute, so the robot side never changes. Use
`--action-space absolute` to train on raw targets.

### Multi-task: one model, prompt selects the task

The instruction string is the only task signal the policy gets, so distinct
phrasings let **one** model do several tasks chosen by the prompt at inference.
Each task has its own phrasing (`TASK_TEMPLATES` in
`convert_microact_to_lerobot.py`): e.g. `targeting` → "move the manipulator toward
the {region} cell"; `patch_clamp` → "record signal from the {region} cell".

Build a **combined** dataset by passing multiple raw sources (one per task) into a
single run — each `ROOT` needs its own `instruction_labels.csv`:

```bash
python dataset_vla/convert_microact_to_lerobot.py \
  --source /data/oocyte_raw:targeting \
  --source /data/patchclamp_raw:patch_clamp \
  --repo-id RaianSilex/MicroVLA_combined --push-to-hub
```

A single-task dataset is just one source (or the back-compatible `--data-root`
`--task patch_clamp`). The two task's **motion orders** (e.g. z→xy vs xy→z) come
straight from each source's demonstrations — nothing to configure. Add a new task
by registering its templates in `TASK_TEMPLATES` and adding a `--source`.

**Resistance + mixed tasks:** if any source carries real `resistance_mohm` (patch
clamp does), the whole dataset gets `observation.resistance`; sources without it
(targeting) get `0`. The policy conditions on it with modality dropout, so it
helps reactive control (contact spike) without becoming the task switch.

> For the prompt to truly select the task, keep visual conditions comparable
> across tasks (ideally some scenes demonstrated under both prompts). If targeting
> and patch-clamp scenes look very different, the model may learn the task from the
> image and ignore the prompt — a data-design issue, not a code one.

## v2.1 (OpenPI / π0 / π0.5)

OpenPI pins an older `lerobot` and needs **v2.1**, not the v3.0 that lerobot ≥ 0.4
writes. The on-disk version is decided by the *installed* `lerobot`, so build the
v2.1 copy in a separate venv:

```bash
python3 -m venv .lerobot-v21-venv && source .lerobot-v21-venv/bin/activate
pip install "lerobot==0.3.3"
python dataset_vla/convert_microact_to_lerobot_v21.py     # -> <repo-id>_v21
```

Same content and region/instruction logic as v3.0; only the on-disk format
differs. The script aborts if the installed `lerobot` would write v3.0.
