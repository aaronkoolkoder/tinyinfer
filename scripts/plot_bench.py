#!/usr/bin/env python3
"""Turns bench/results/compare.csv into a markdown summary and charts."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = Path(sys.argv[1]), Path(sys.argv[2])
data = defaultdict(dict)  # (engine, test) -> {threads: (tps, sd)}
for r in csv.DictReader(open(src)):
    data[(r["engine"], r["test"])][int(r["threads"])] = (float(r["tps"]), float(r["sd"]))

engines = ["tinyinfer", "llama.cpp"]
tests = sorted({t for _, t in data}, key=lambda t: (t[:2] != "pp", t))
colors = {"tinyinfer": "#2a6fdb", "llama.cpp": "#9aa0a6"}
titles = {"pp": "Prefill (prompt processing)", "tg": "Decode (token generation)"}

lines = ["| test | tinyinfer best | llama.cpp best | ratio |", "|---|---:|---:|---:|"]
for test in tests:
    best = {}
    for e in engines:
        runs = data.get((e, test), {})
        if runs:
            t, (tps, _) = max(runs.items(), key=lambda kv: kv[1][0])
            best[e] = (tps, t)
    if len(best) == 2:
        ours, theirs = best["tinyinfer"], best["llama.cpp"]
        lines.append(f"| {test} | {ours[0]:.1f} tok/s ({ours[1]} threads) | {theirs[0]:.1f} tok/s ({theirs[1]} threads) "
                     f"| **{100 * ours[0] / theirs[0]:.0f}%** |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, axes = plt.subplots(1, len(tests), figsize=(5.5 * len(tests), 4))
axes = axes if len(tests) > 1 else [axes]
for ax, test in zip(axes, tests):
    threads = sorted({t for e in engines for t in data.get((e, test), {})})
    width = 0.38
    for i, e in enumerate(engines):
        runs = data.get((e, test), {})
        xs = [j + (i - 0.5) * width for j in range(len(threads))]
        ys = [runs.get(t, (0, 0))[0] for t in threads]
        errs = [runs.get(t, (0, 0))[1] for t in threads]
        ax.bar(xs, ys, width, yerr=errs, label=e, color=colors[e], capsize=3)
    ax.set_xticks(range(len(threads)), [str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel("tokens / second")
    ax.set_title(f"{titles.get(test[:2], test)}: {test}")
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
fig.suptitle("Qwen2.5-0.5B-Instruct, F32, CPU only (Apple M5 Pro)")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(out / "compare.png", dpi=150)
print(f"wrote {out / 'compare.png'}")
