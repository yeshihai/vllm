"""Summarize a P/D bench result directory: TTFT/ITL/throughput + HiSparse miss rate."""

import glob
import json
import os
import re
import sys


def point_names(log: str) -> list[str]:
    names: list[str] = []
    with open(log, errors="ignore") as f:
        for line in f:
            m = re.match(r"### (\S+):", line)
            if m:
                names.append(m.group(1))
    return names


def hisparse_totals(log: str) -> dict[str, tuple[float, float, float]]:
    first: dict[tuple[str, str], float] = {}
    last: dict[tuple[str, str], float] = {}
    point = None
    pat = re.compile(r"METRIC (vllm:hisparse_\w+)\{.*\} ([0-9.eE+]+)")
    with open(log, errors="ignore") as f:
        for line in f:
            m = re.match(r"### (\S+):", line)
            if m:
                point = m.group(1)
            m = pat.search(line)
            if m and point:
                key = (point, m.group(1))
                first.setdefault(key, float(m.group(2)))
                last[key] = float(m.group(2))
    totals: dict[str, tuple[float, float, float]] = {}
    for point in {p for p, _ in first}:
        hits = last.get((point, "vllm:hisparse_cache_hits_total"), 0.0) - first.get(
            (point, "vllm:hisparse_cache_hits_total"), 0.0
        )
        misses = last.get((point, "vllm:hisparse_cache_misses_total"), 0.0) - first.get(
            (point, "vllm:hisparse_cache_misses_total"), 0.0
        )
        h2d = last.get(
            (point, "vllm:hisparse_host_to_device_bytes_total"), 0.0
        ) - first.get((point, "vllm:hisparse_host_to_device_bytes_total"), 0.0)
        totals[point] = (hits, misses, h2d)
    return totals


def main() -> None:
    d = sys.argv[1]
    files = sorted(
        glob.glob(os.path.join(d, "*.json")),
        key=lambda p: re.search(r"-(\d{8}-\d{6})\.json$", p).group(1),
    )
    names = point_names(os.path.join(d, "run.log"))
    if len(files) == len(names) + 1:
        files = files[1:]
    print(
        f"{'point':28s} {'TTFT50':>8s} {'TTFT90':>8s} {'ITL50':>8s} "
        f"{'ITL90':>8s} {'tok/s':>9s} {'req/s':>7s}"
    )
    for name, path in zip(names, files, strict=False):
        with open(path) as f:
            j = json.load(f)
        print(
            f"{name:28s} {j.get('median_ttft_ms', 0):8.1f} "
            f"{j.get('p90_ttft_ms', 0):8.1f} {j.get('median_itl_ms', 0):8.2f} "
            f"{j.get('p90_itl_ms', 0):8.2f} {j.get('output_throughput', 0):9.1f} "
            f"{j.get('request_throughput', 0):7.2f}"
        )
    totals = hisparse_totals(os.path.join(d, "run.log"))
    if not totals:
        return
    print(f"\n{'point':28s} {'hits':>12s} {'misses':>12s} {'miss%':>7s} {'H2D GB':>9s}")
    for point in sorted(totals):
        hits, misses, h2d = totals[point]
        rate = 100 * misses / (hits + misses) if hits + misses else 0.0
        print(f"{point:28s} {hits:12.0f} {misses:12.0f} {rate:6.2f}% {h2d / 1e9:8.2f}")


if __name__ == "__main__":
    main()
