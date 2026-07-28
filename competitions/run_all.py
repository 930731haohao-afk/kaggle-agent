"""Batch-run the generic pipeline over multiple competitions; robust to per-comp errors."""
import json
import subprocess
import sys
import time

COMPS = sys.argv[1:] or [f"playground-series-s3e{e}" for e in (1, 3, 5, 7, 9, 11, 14, 19)]
results = []
for c in COMPS:
    print(f"\n{'='*70}\n>>> {c}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        out = subprocess.run([sys.executable, "competitions/run_competition.py", c],
                             capture_output=True, text=True, timeout=1800)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
        for l in out.stdout.splitlines():
            if not (l.strip().startswith(("[LightGBM]",)) or "UserWarning" in l or "warnings.warn" in l):
                print(l, flush=True)
        if line:
            r = json.loads(line[0][7:])
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)
        else:
            print("STDERR tail:", out.stderr.strip().splitlines()[-5:], flush=True)
            results.append(dict(comp=c, error=(out.stderr.strip().splitlines() or ["?"])[-1]))
    except Exception as e:
        results.append(dict(comp=c, error=str(e)))
        print("EXCEPTION", e, flush=True)

print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
for r in results:
    if "error" in r:
        print(f"  {r['comp']:28} ERROR: {r['error'][:60]}", flush=True)
    else:
        print(f"  {r['comp']:28} {r['metric']:6} = {r['score']:<10} weights={r['weights']} ({r.get('time_s','?')}s)", flush=True)
json.dump(results, open("competitions/_batch_results.json", "w"), indent=2)
print("\nwrote competitions/_batch_results.json", flush=True)
