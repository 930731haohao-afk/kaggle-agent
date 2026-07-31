#!/usr/bin/env python3
"""Build clean, agent-neutral competition roots.

Why this exists: the benchmark harness used to hand AIDE the whole of
`kaggle/competitions/<comp>/data/`, which is my own agent's working directory. Wherever my
pipeline had cached processed feature tables, tuned-parameter JSONs or OOF arrays, AIDE
received them as if they were competition data and — reasonably — used them. Four runs
(s3e1, s3e5, s3e19, s6e7) are void as a result.

The fix is structural: every agent reads from `bench-comps/<comp>/`, which holds only
`config.yaml` plus a `data/` view of **official competition files only**. No agent can see
another's intermediates because they are not there to see.

Officialness comes from Kaggle's own file manifest (`competitions files -c <comp>`), cached
to `manifests/<comp>.json`. A filename allow-list was tried first and rejected: it wrongly
excluded `gender_submission.csv`, store-sales' `oil.csv`/`stores.csv`/`transactions.csv`,
and the prosumers auxiliary tables — all genuinely official. Only the manifest can tell an
official auxiliary table from another agent's cache.

Usage:
    python build_isolated_roots.py <comp> [...]           # dry run
    python build_isolated_roots.py --write <comp> [...]   # build
    python build_isolated_roots.py --write --all          # every comp with a data/ dir
"""

import json
import os
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path

SRC = Path.home() / "ai_agents/kaggle/competitions"
DST = Path.home() / "ai_agents/bench-comps"
MANIFESTS = DST / "manifests"
KAGGLE = Path.home() / "ai_agents/mle-bench/.venv/bin/kaggle"
TOKEN = Path.home() / ".kaggle/huang_token"


def official_files(comp: str) -> set[str] | None:
    """Kaggle's declared file list for a competition, cached. None if unavailable."""
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    cache = MANIFESTS / f"{comp}.json"
    if cache.is_file():
        return set(json.loads(cache.read_text()))

    env = dict(os.environ, KAGGLE_API_TOKEN=TOKEN.read_text().strip())
    try:
        out = subprocess.run(
            [str(KAGGLE), "competitions", "files", "-c", comp, "--csv"],
            capture_output=True, text=True, timeout=120, env=env, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  ! {comp}: manifest fetch failed ({e.__class__.__name__})")
        return None

    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if len(lines) < 2 or not lines[0].lower().startswith("name"):
        print(f"  ! {comp}: no manifest ({(out.stdout or out.stderr).strip()[:80]})")
        return None

    names = {ln.split(",")[0].strip() for ln in lines[1:] if ln.split(",")[0].strip()}
    cache.write_text(json.dumps(sorted(names), indent=1))
    return names


def build(comp: str, write: bool) -> tuple[int, int]:
    data = SRC / comp / "data"
    if not data.is_dir():
        return 0, 0

    manifest = official_files(comp)
    if manifest is None:
        print(f"{comp}: SKIPPED — cannot verify officialness without a manifest")
        return 0, 0

    present = [p for p in data.iterdir() if p.is_file()]

    # Some competitions distribute archives (`train.csv.zip`, `train.zip`), and the local
    # copy holds the extracted member alongside it. The member is official content — it just
    # isn't named in the manifest. Read each official archive's index and treat its members
    # as official too, rather than handing agents only zips and forcing each one to unpack
    # differently. Verified by listing the archive, not by guessing from the filename.
    archive_members: set[str] = set()
    for p in present:
        if p.suffix == ".zip" and (p.name in manifest or p.name == f"{comp}.zip"):
            try:
                with zipfile.ZipFile(p) as z:
                    archive_members.update(Path(n).name for n in z.namelist() if not n.endswith("/"))
            except (zipfile.BadZipFile, OSError) as e:
                print(f"  ! {comp}/{p.name}: cannot read archive index ({e.__class__.__name__})")

    def official_name(name: str) -> bool:
        return name in manifest or name == f"{comp}.zip" or name in archive_members

    official = [p for p in present if official_name(p.name)]
    excluded = [p for p in present if p not in official]

    note = f"  <-- EXCLUDES {[p.name for p in excluded]}" if excluded else ""
    print(f"{comp}: {len(official)} official / {len(excluded)} excluded{note}")

    missing = manifest - {p.name for p in present}
    if missing:
        print(f"  ! {comp}: manifest lists files not present locally: {sorted(missing)}")

    if write:
        out_data = DST / comp / "data"
        # If `data` is already a symlink to a prepared dataset (the MLE-bench comps point
        # at ~/.cache/mle-bench/.../prepared/public), mkdir(exist_ok=True) follows it and we
        # would write links *into the source itself* — each link then targets its own path,
        # a self-referential loop that destroys the file. This happened on 2026-07-27 to
        # us-patent and ventilator; restored from workspace copies, checksums verified.
        # Such a root is already isolated by construction, so leave it alone.
        if out_data.is_symlink():
            print(f"  = {comp}: data/ is an existing symlinked dataset — left untouched")
            return len(official), len(excluded)
        out_data.mkdir(parents=True, exist_ok=True)
        cfg = SRC / comp / "config.yaml"
        if cfg.is_file():
            shutil.copy2(cfg, DST / comp / "config.yaml")
        # Symlink, not copy: official data is large and immutable, and a link makes it
        # plain that this root is a view rather than a second source of truth.
        for p in official:
            link = out_data / p.name
            target = p.resolve()
            # Second guard: never make a link that points at itself.
            if link.resolve() == target and link.is_symlink():
                continue
            if target == link:
                print(f"  ! {comp}/{p.name}: source resolves to the link path — skipped")
                continue
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)

    return len(official), len(excluded)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    if "--all" in sys.argv:
        args = sorted(d.name for d in SRC.iterdir() if (d / "data").is_dir())
    if not args:
        sys.exit("usage: build_isolated_roots.py [--write] <comp>... | --all")

    kept = dropped = 0
    for comp in args:
        k, d = build(comp, write)
        kept += k
        dropped += d

    print(f"\n{'BUILT' if write else 'DRY RUN'}: {kept} official linked, {dropped} excluded")
    if not write:
        print("re-run with --write to build")


if __name__ == "__main__":
    main()
