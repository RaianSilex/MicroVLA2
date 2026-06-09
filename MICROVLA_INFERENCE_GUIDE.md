# MicroVLA Inference (local rollout) — checkpoint → robot

Minimal, copy-paste recipe for running a trained MicroVLA checkpoint on the Sensapex rig.

Unlike pi0/OpenPI (server + client), MicroVLA inference is **all local**: `rollout/vla_main.py`
loads the checkpoint and talks **directly** to the ump_suite ROS topics. The checkpoint is
**self-contained** — it bundles the norm stats, vocabs, **and** the backbone/language config — so the
**same command runs every backbone** (dinov2+cellpose4 / resnet18 / resnet18+cellpose4); only
`--checkpoint` changes. It auto-detects the rest from the `.pt`.

> **Placeholders to change before pasting:**
> - `chowd207@login.msi.umn.edu` → your MSI login
> - `/projects/standard/suhasabk/shared` → your MSI project dir

---

## 0. The rules that bit us

| Rule | Why |
|---|---|
| **One Python needs ROS *and* torch** | `rollout/sensapex_env.py` uses `rclpy`; the policy needs `torch`+`transformers`(+`cellpose`). Run the repo `.venv` with ROS **sourced** (both are py3.10, so `rclpy` resolves via `PYTHONPATH`). |
| **Use an in-distribution instruction** | The model was trained on templated, region-grounded prompts (§4). A random prompt → off-distribution behavior. |
| **`--dry-run` first** | Loads + infers + writes the preview **without** commanding the motors. Always validate a checkpoint this way before live motion. |
| **Use `vla_policy_best.pt`** | Lowest val loss. `last.pt` is just the most recent epoch. |

---

## 1. Copy the checkpoint(s) from MSI

Pull `vla_policy_best.pt` from each run dir into the matching local dir:

```bash
for d in checkpoints_vla_resnet_165 checkpoints_vla_cp4_165beads_new checkpoints_vla_resnet_cp4_165; do
  mkdir -p ~/MicroVLA/$d
  rsync -avhP chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/$d/vla_policy_best.pt ~/MicroVLA/$d/
done
```
> If a run never beat its initial val, there's no `best.pt` — use `vla_policy_last.pt`. The checkpoint
> is everything you need; there is **no** separate stats file for MicroVLA.

---

## 2. One-time: rollout deps in `.venv`

The repo `.venv` (py3.10, torch+CUDA) needs two additions:

```bash
cd ~/MicroVLA
.venv/bin/pip install "transformers==4.49.0"     # ALL checkpoints (DistilBERT language backend)
.venv/bin/pip install "cellpose>=4.0"            # ONLY for the two +cellpose4 checkpoints
```
- Pin `transformers==4.49.0` so it stays compatible with the lerobot install in the same `.venv`.
- `cellpose` may want `numpy<2`; if it conflicts with this `.venv`, give it its own venv instead of
  downgrading. **The `resnet18` checkpoint needs no cellpose** — start there.
- Backbone weights (DINOv2 / Cellpose-SAM / ResNet) download automatically on first run (local has internet).

---

## 3. Run it — 2 terminals

**Terminal 1 — robot + camera:**
```bash
cd ~/ros2_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch ump_suite app.launch.py
```

**Terminal 2 — the rollout (dry-run first, no motion):**
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/MicroVLA
.venv/bin/python -m rollout.vla_main \
  --checkpoint checkpoints_vla_resnet_165/vla_policy_best.pt \   # CHANGE per run (§4)
  --instruction "move both manipulators toward the center cell" \  # CHANGE to match the bead (§4)
  --dry-run
```
On load you'll see `[microvla] loaded … (epoch=…, backbone=…, language=…, device=cuda)`. If it infers
and the preview `microact_vla_live.png` looks right, **drop `--dry-run`** to command the robot.

---

## 4. Change the checkpoint / instruction

- **Checkpoint:** set `--checkpoint` to any of:
  - `checkpoints_vla_resnet_165/vla_policy_best.pt` (resnet18)
  - `checkpoints_vla_cp4_165beads_new/vla_policy_best.pt` (dinov2+cellpose4)
  - `checkpoints_vla_resnet_cp4_165/vla_policy_best.pt` (resnet18+cellpose4)

  The backbone, language backend, and action space are read from the checkpoint — **nothing else to change.**

- **Instruction (keep it in-distribution):** pick one template and set `{R}` to where the bead is in
  the frame:

  | Template |
  |---|
  | `move both manipulators toward the {R} cell` |
  | `advance both needles to the {R} cell` |
  | `target the {R} cell with both manipulators` |
  | `guide the pipettes to the cell in the {R}` |
  | `bring the two pipettes to the {R} cell` |

  `{R}` ∈ `center`/`middle`, `top-left`/`upper-left`, `top`, `top-right`, `left`/`middle-left`,
  `right`, `bottom-left`/`lower-left`, `bottom`, `bottom-right`/`lower-right`.
  e.g. `advance both needles to the top-left cell`.

Useful flags: `--device cpu`, `--max-timesteps`, `--open-loop-horizon`, `--temporal-agg` /
`--no-temporal-agg`, `--default-speed`, `--preview-path`.

---

## 5. Safety & gotchas

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: rclpy` | source ROS **before** running, and use the `.venv` python (py3.10 matches Humble). If still failing, recreate the venv with `python3.10 -m venv --system-site-packages`. |
| `ModuleNotFoundError: transformers` / `cellpose` | install per §2 (`cellpose` only for `+cellpose4`). |
| No camera/state / hangs waiting | Terminal 1 `app.launch.py` not up, or topics not publishing (`ros2 topic echo /ump/live`). |
| Robot moves unexpectedly | you dropped `--dry-run` — re-check the per-axis limits / step caps in `rollout/adapters/sensapex_dual.py` + `rollout/sensapex_env.py` match your workspace. |
| Policy "ignores the image" | known posterior-collapse tendency; compare checkpoints and use `vla_policy_best.pt`. |

> **Before live motion:** the per-axis workspace bounds and per-tick step caps live in the adapter
> (`rollout/adapters/sensapex_dual.py` / `rollout/sensapex_env.py`). `--dry-run` exercises everything
> except commanding the motors — use it to confirm the checkpoint produces sane targets first.