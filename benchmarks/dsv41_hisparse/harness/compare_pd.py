"""Compare two P/D bench result directories point by point."""

import glob
import json
import os
import re
import sys


def _timestamp(path: str) -> str:
    match = re.search(r"-(\d{8}-\d{6})\.json$", path)
    assert match is not None
    return match.group(1)


def load(d: str) -> dict[str, dict]:
    log = os.path.join(d, "run.log")
    names: list[str] = []
    with open(log, errors="ignore") as f:
        for line in f:
            match = re.match(r"### (\S+):", line)
            if match:
                names.append(match.group(1))
    files = sorted(glob.glob(os.path.join(d, "*.json")), key=_timestamp)
    if len(files) == len(names) + 1:
        # The kernel warmup point logs without a colon and is not measured.
        files = files[1:]
    if len(files) != len(names):
        print(f"WARN {d}: {len(files)} json vs {len(names)} points")
    points: dict[str, dict] = {}
    for name, path in zip(names, files, strict=False):
        with open(path) as f:
            points[name] = json.load(f)

    first: dict[tuple[str, str], float] = {}
    last: dict[tuple[str, str], float] = {}
    point = None
    pat = re.compile(r"METRIC (vllm:hisparse_\w+)\{.*\} ([0-9.eE+]+)")
    with open(log, errors="ignore") as f:
        for line in f:
            match = re.match(r"### (\S+):", line)
            if match:
                point = match.group(1)
            match = pat.search(line)
            if match and point:
                key = (point, match.group(1))
                first.setdefault(key, float(match.group(2)))
                last[key] = float(match.group(2))
    for name in points:
        hits = last.get((name, "vllm:hisparse_cache_hits_total"), 0.0) - first.get(
            (name, "vllm:hisparse_cache_hits_total"), 0.0
        )
        misses = last.get((name, "vllm:hisparse_cache_misses_total"), 0.0) - first.get(
            (name, "vllm:hisparse_cache_misses_total"), 0.0
        )
        h2d = last.get(
            (name, "vllm:hisparse_host_to_device_bytes_total"), 0.0
        ) - first.get((name, "vllm:hisparse_host_to_device_bytes_total"), 0.0)
        points[name]["_miss_rate"] = (
            100 * misses / (hits + misses) if hits + misses else 0.0
        )
        points[name]["_h2d_gb"] = h2d / 1e9
    return points


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    label_a = sys.argv[3] if len(sys.argv) > 3 else "A"
    label_b = sys.argv[4] if len(sys.argv) > 4 else "B"
    print(
        f"{'point':18s} | {label_a + ' TTFT':>10s} {label_b + ' TTFT':>10s} "
        f"{'dTTFT%':>9s} | {label_a + ' ITL':>10s} {label_b + ' ITL':>10s} "
        f"{'dITL%':>8s} | {'miss%':>8s} {'H2D GB':>8s}"
    )
    for name in a:
        if name == "warmup" or name not in b:
            continue
        x, y = a[name], b[name]
        ttft_a = x.get("median_ttft_ms", 0.0)
        ttft_b = y.get("median_ttft_ms", 0.0)
        itl_a = x.get("median_itl_ms", 0.0)
        itl_b = y.get("median_itl_ms", 0.0)
        d_ttft = 100 * (ttft_b - ttft_a) / ttft_a if ttft_a else 0.0
        d_itl = 100 * (itl_b - itl_a) / itl_a if itl_a else 0.0
        print(
            f"{name:18s} | {ttft_a:10.1f} {ttft_b:10.1f} {d_ttft:8.1f}% | "
            f"{itl_a:10.2f} {itl_b:10.2f} {d_itl:7.1f}% | "
            f"{y.get('_miss_rate', 0):7.2f}% {y.get('_h2d_gb', 0):8.2f}"
        )


if __name__ == "__main__":
    main()
