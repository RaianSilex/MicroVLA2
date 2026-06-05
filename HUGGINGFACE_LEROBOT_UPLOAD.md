# Upload raw frames + CSV → HuggingFace (LeRobot v2.1 and v3)

Converts the raw `dataset/` (camera frames + `instruction_labels.csv`) into a LeRobot dataset and
pushes it to HuggingFace. Run this **on your laptop** (where the raw data lives), not on MSI.

- **v3** (lerobot 0.4.x) → for **MicroVLA** training. See [MSI_MICROVLA_TRAINING_GUIDE.md](MSI_MICROVLA_TRAINING_GUIDE.md).
- **v2.1** (lerobot 0.3.3) → for **OpenPI / pi0**. See [bsbrl-openpi docs](../bsbrl-openpi/docs/MSI_PI0_SENSAPEX_LOWMEM_FINETUNE.md).

> **Placeholders to change before pasting:**
> - `RaianSilex/multibeads_165episodes` → your target HF dataset repo id (v3)
> - `RaianSilex/multibead_165episodes_lerobotv21` → your target HF dataset repo id (v2.1)

---

## What goes in

```
dataset/
├── instruction_labels.csv      # columns: trial_id,region,instruction
└── saved_frames/<trial>/...    # one folder per trial (episode)
```
Each row of the CSV = one episode; `region` is the bead location, `instruction` the language label.

---

## 0. HuggingFace auth (do this once, never hard-code a token)

The converters push **private** datasets, so you need a **WRITE** token:
```bash
huggingface-cli login          # paste a WRITE token (input is hidden). NEVER put a token in a file/script/doc.
```
Create the empty dataset repos first (or let `--push-to-hub` create them):
`https://huggingface.co/new-dataset` → name them to match your `--repo-id`.

---

## 1. Pick the right environment

There are **two** venvs because v2.1 and v3 need different `lerobot` versions:

| Format | venv | key pins |
|---|---|---|
| v3 | `.venv` | `lerobot==0.4.4`, `transformers==4.49.0`, python 3.10 |
| v2.1 | `.lerobot-v21-venv` | `lerobot==0.3.3`, python 3.10 |

<details><summary>First-time: create the venvs (skip if they already exist)</summary>

```bash
# v3
python3.10 -m venv .venv && source .venv/bin/activate
pip install "lerobot==0.4.4" "transformers==4.49.0"
deactivate
# v2.1
python3.10 -m venv .lerobot-v21-venv && source .lerobot-v21-venv/bin/activate
pip install "lerobot==0.3.3"
deactivate
```
> The v3 pin matters: newer `transformers` needs `huggingface-hub>=1.x`, which `lerobot 0.4.4` caps —
> `transformers==4.49.0` is the version that resolves cleanly. Use **python 3.10** (lerobot needs ≥3.10).
</details>

---

## 2. Convert + push — v3 (for MicroVLA)

```bash
cd ~/MicroVLA
source .venv/bin/activate
python dataset_vla/convert_microact_to_lerobot.py \
  --repo-id RaianSilex/multibeads_165episodes \      # CHANGE
  --push-to-hub
deactivate
```

## 3. Convert + push — v2.1 (for OpenPI)

```bash
cd ~/MicroVLA
source .lerobot-v21-venv/bin/activate
python dataset_vla/convert_microact_to_lerobot_v21.py \
  --repo-id RaianSilex/multibead_165episodes_lerobotv21 \   # CHANGE
  --push-to-hub
deactivate
```

> **Both at once?** Yes — run §2 and §3 in two separate terminals. They use different venvs and
> different `--repo-id`s, so they don't collide. (Without `--push-to-hub` they only build locally.)

---

## 4. Defaults you usually don't change

Both converters share these (`--flag default`):
`--data-root dataset/` · `--labels dataset/instruction_labels.csv` ·
`--robot-type sensapex_dual_ump4` · `--fps 3` · `--down-h 540 --down-w 720` ·
`--limit-trials 0` (0 = all; set e.g. `--limit-trials 5` for a quick smoke test) · pushes `private=True`.

To point at different raw data: `--data-root <dir> --labels <dir>/instruction_labels.csv`.
To change the robot id stored in the dataset (must match training config): `--robot-type <name>`.

---

## 5. Verify

```bash
huggingface-cli repo info --repo-type dataset RaianSilex/multibeads_165episodes   # CHANGE
```
or open `https://huggingface.co/datasets/<your-repo-id>`. The training guides download it by repo id
on first run (private datasets need a **read** token via `huggingface-cli login` on MSI).

---

## 6. Gotchas

| Symptom | Fix |
|---|---|
| `ResolutionImpossible` installing v3 deps | use the pins in §1 (`lerobot==0.4.4` + `transformers==4.49.0`) |
| `Defaulting to user installation` / python 3.8 | venv not active or wrong python; recreate with **python 3.10** |
| `401`/`403` on push | not logged in, or a **read** token — `huggingface-cli login` with a **write** token |
| v2.1 dataset rejected by OpenPI | OpenPI needs **v2.1** (`.lerobot-v21-venv` / lerobot 0.3.3), not v3 |
