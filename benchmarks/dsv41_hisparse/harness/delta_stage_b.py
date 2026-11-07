import glob
import json
import math
import os
import re

BASE = "/dockerdata/yeshihai_dsv41_buffer/results/stage_b_20261107_r1"
WORKLOADS = [
    "b_131k_c1",
    "b_262k_c1",
    "b_524k_c1",
    "b_1m_c1",
    "b_131k_c8",
    "b_131k_c16",
    "b_131k_c32",
    "b_4k_c32",
    "d_c1",
    "d_c4",
    "d_c8",
    "d_c16",
    "d_c32",
]


def result(mode, work):
    files = glob.glob(os.path.join(BASE, mode, work, "*.json"))
    if not files:
        return None
    d = json.load(open(files[0]))
    return d


def cap(mode):
    text = open(os.path.join(BASE, f"capacity_{mode}.txt")).read()
    m = re.search(r"([0-9,]+) tokens.*?([0-9.]+)x", text)
    return int(m.group(1).replace(",", "")), float(m.group(2))


print("workload\tmetric\ton\toff\tdelta_pct")
for work in WORKLOADS:
    on, off = result("on", work), result("off", work)
    if on is None or off is None:
        print(f"{work}\tMISSING")
        continue
    for label, key in [
        ("ttft_ms", "mean_ttft_ms"),
        ("ttft_p99_ms", "p99_ttft_ms"),
        ("itl_ms", "median_itl_ms"),
        ("itl_p99_ms", "p99_itl_ms"),
        ("out_tput", "output_throughput"),
    ]:
        a, b = on.get(key), off.get(key)
        if a is None or b is None:
            continue
        delta = (a - b) / b * 100 if b else 0.0
        print(f"{work}\t{label}\t{a:.4f}\t{b:.4f}\t{delta:+.2f}%")

for mode in ["on", "off"]:
    print(f"capacity_{mode}\t{cap(mode)[0]}\t{cap(mode)[1]:.2f}x")
print(f"capacity_ratio\t{cap('on')[1] / cap('off')[1]:.2f}x")

on_rows = json.load(open(f"/dockerdata/yeshihai_dsv41_buffer/results/dsv41_gsm8k_stage_b_on_4096_full1319.json"))
off_rows = json.load(open(f"/dockerdata/yeshihai_dsv41_buffer/results/dsv41_gsm8k_stage_b_off_4096_full1319.json"))
on_map = {r["question"]: r["correct"] for r in on_rows}
off_map = {r["question"]: r["correct"] for r in off_rows}
both = [q for q in on_map if q in off_map]
b = sum(on_map[q] and not off_map[q] for q in both)
c = sum(not on_map[q] and off_map[q] for q in both)
n = b + c
p = min(1.0, 2 * sum(math.comb(n, k) * 0.5**n for k in range(0, min(b, c) + 1))) if n else 1.0
print(f"GSM8K on={sum(on_map[q] for q in both)}/{len(both)} off={sum(off_map[q] for q in both)}/{len(both)} "
      f"b={b} c={c} mcnemar_p={p:.3f}")
