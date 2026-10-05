"""Download complete SoccerNet 720p matches to the external data drive.

Uses the locally authenticated Hugging Face account. Only the 720p halves
(`1_720p.mkv`, `2_720p.mkv`) are fetched. Rerunning reuses files already
completed by huggingface_hub; no token is written to the output folder.

Examples (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\download_soccernet_720p.py
  .\\.venv\\Scripts\\python.exe scripts\\download_soccernet_720p.py --sample 20 --seed 7 --download
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))

REPO = "SoccerNet/SoccerNet_raw_HQ"
BRANCH = "videos-720p"
MATCHES = (
    "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
    "spain_laliga/2015-2016/2015-08-29 - 21-30 Barcelona 1 - 0 Malaga",
    "europe_uefa-champions-league/2016-2017/2016-09-13 - 21-45 Barcelona 7 - 0 Celtic",
)
# Share of a stratified sample per official split; leagues are then drawn in
# proportion to their availability inside each split.
SPLIT_SHARE = {"train": .6, "val": .2, "test": .2}


def stratified_sample(candidates, splits, count, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for game in sorted(candidates):
        strata[(splits.get(game, "unknown"), game.split("/")[0])].append(game)
    chosen = []
    for split, share in SPLIT_SHARE.items():
        target = round(count * share)
        leagues = {league: games for (s, league), games in strata.items() if s == split}
        total = sum(len(g) for g in leagues.values())
        if not total:
            continue
        quota = {league: len(games) / total * target for league, games in leagues.items()}
        picks = {league: int(q) for league, q in quota.items()}
        # Largest remainders fill the rounding gap, so each split hits its target.
        for league in sorted(quota, key=lambda k: quota[k] - picks[k], reverse=True)[:target - sum(picks.values())]:
            picks[league] += 1
        for league, n in picks.items():
            chosen += rng.sample(leagues[league], min(n, len(leagues[league])))
    return sorted(chosen)


def main():
    from huggingface_hub import HfApi, get_token, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=Path(r"D:\CVDL Football Data\SoccerNet\videos-720p"))
    parser.add_argument("--games", nargs="*", default=None, help="Exact league/season/game paths")
    parser.add_argument("--sample", type=int, default=0,
                        help="Add a stratified sample of this many games not already downloaded")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download", action="store_true", help="Download the selected half videos")
    args = parser.parse_args()
    token = get_token()
    if not token:
        parser.error("Log in locally with the approved Hugging Face account first")
    info = HfApi(token=token).dataset_info(REPO, revision=BRANCH, files_metadata=True)
    available = {item.rfilename: item for item in info.siblings}
    games_online = {name.rsplit("/", 1)[0] for name in available if name.endswith("_720p.mkv")}
    complete = {g for g in games_online if all(f"{g}/{h}_720p.mkv" in available for h in (1, 2))}
    inventory_path = args.output / "pitchprofile_download.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    local = set(inventory.get("matches", []))
    if args.sample:
        splits_path = args.output.parent / "official_splits.json"
        if not splits_path.is_file():
            parser.error(f"Run scripts/sync_soccernet_labels.py first; {splits_path} is missing")
        splits = json.loads(splits_path.read_text(encoding="utf-8"))["games"]
        matches = stratified_sample(complete - local, splits, args.sample, args.seed)
    else:
        matches = list(args.games or MATCHES)
    missing = [g for g in matches if g not in complete]
    if missing:
        raise RuntimeError(f"Not available with both 720p halves on {BRANCH}: {missing}")
    selected = [f"{match}/{half}_720p.mkv" for match in matches for half in (1, 2)]
    total = sum(available[name].size for name in selected)
    print(f"{len(matches)} matches, {len(selected)} half videos, {total / 1e9:.2f} GB", flush=True)
    for name in selected:
        print(f"{available[name].size / 1e9:.2f} GB  {name}", flush=True)
    if not args.download:
        print("Add --download to acquire these files.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] Downloading {name}", flush=True)
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=info.sha,
                                    token=token, local_dir=args.output))
        if path.stat().st_size != available[name].size:
            raise RuntimeError(f"Downloaded size does not match repository metadata: {name}")
        print(f"[{index}/{len(selected)}] Verified {path}", flush=True)
        # Record each completed match as soon as both halves exist, so an
        # interrupted run still leaves an accurate inventory.
        match = name.rsplit("/", 1)[0]
        if all((args.output / f"{match}/{h}_720p.mkv").is_file() for h in (1, 2)) and match not in local:
            local.add(match)
            files = inventory.setdefault("files", [])
            files += [{"path": f"{match}/{h}_720p.mkv", "bytes": available[f"{match}/{h}_720p.mkv"].size} for h in (1, 2)]
            inventory.update(repository=REPO, branch=BRANCH, revision=info.sha, matches=sorted(local))
            inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Completed {len(selected)} video halves in {args.output}", flush=True)


if __name__ == "__main__":
    main()
