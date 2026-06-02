"""Upload a local folder (a trained checkpoint or a LeRobot dataset) to the Hugging Face Hub.

Salvaged from the old OpenPI rollout tooling and rewritten to use argparse (no
`tyro`) so it runs with MicroVLA's own dependencies. `huggingface_hub` is pulled
in by the optional `lerobot` extra (see requirements.txt).

The dataset converter (`dataset_vla/convert_microact_to_lerobot.py`) already pushes
the dataset when `--push-to-hub` is set, so this script is mainly useful for
pushing a *trained checkpoint* directory to the Hub after training.

Examples:
    # Push a trained MicroVLA checkpoint directory (a model repo)
    python push_to_huggingface.py \
        --local-dir checkpoints/microvla_dinov2cp4_66episodes_100epochs \
        --repo-id <user>/microvla_dinov2cp4 \
        --repo-type model

    # Push a LeRobot dataset directory (a dataset repo)
    python push_to_huggingface.py \
        --local-dir ~/.cache/huggingface/lerobot/<user>/microvla_ump_dataset \
        --repo-id <user>/microvla_ump_dataset \
        --repo-type dataset
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload a local folder (checkpoint or LeRobot dataset) to the Hugging Face Hub."
    )
    p.add_argument("--local-dir", required=True,
                   help="Local folder to upload (a checkpoint dir or a LeRobot dataset dir).")
    p.add_argument("--repo-id", required=True, help="Destination repo on the Hub, e.g. user/name.")
    p.add_argument("--repo-type", choices=("model", "dataset"), default="model",
                   help="'model' for checkpoints, 'dataset' for LeRobot datasets.")
    p.add_argument("--private", action="store_true", help="Create the repo as private.")
    p.add_argument("--commit-message", default="Upload from MicroVLA")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import so the rest of the repo doesn't require huggingface_hub.
    from huggingface_hub import HfApi

    api = HfApi()
    # Create the repo if it does not exist.
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    # Upload the folder contents to the repo.
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.local_dir} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
