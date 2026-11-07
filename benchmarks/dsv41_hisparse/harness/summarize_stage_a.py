import glob
import json
import os
import re

BASE = "/dockerdata/yeshihai_dsv41_buffer/results/stage_a_20261107_r1"
METRICS = os.path.join(BASE, "metrics.log")
WORKLOADS = ["a_1m_c1", "a_131k_c16", "a_131k_c32"]


def result(mode, mbt, work):
    files = glob.glob(os.path.join(BASE, mode, str(mbt), work, "*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        d = json.load(f)
    return {
        "ttft": d.get("mean_ttft_ms"),
        "ttft_p99": d.get("p99_ttft_ms"),
        "itl": d.get("median_itl_ms"),
        "itl_p99": d.get("p99_itl_ms"),
        "out_tput": d.get("output_throughput"),
        "completed": d.get("completed"),
        "failed": d.get("failed"),
    }


def capacity(mode, mbt):
    path = os.path.join(BASE, f"capacity_{mode}_{mbt}.txt")
    if not os.path.exists(path):
        return None, None
    text = open(path).read()
    m = re.search(r"([0-9,]+) tokens.*?([0-9.]+)x", text)
    if not m:
        return None, None
    return int(m.group(1).replace(",", "")), float(m.group(2))


def metric_deltas(mode, mbt, work):
    if mode != "on" or not os.path.exists(METRICS):
        return None
    vals = {"before": {}, "after": {}}
    pat = re.compile(
        r"\] (on|off) mbt=(\d+) (\S+) (before|after) "
        r"vllm:hisparse_cache_hits_total\{[^}]*\} ([0-9.eE+]+) "
        r"vllm:hisparse_cache_misses_total\{[^}]*\} ([0-9.eE+]+) "
        r"vllm:hisparse_host_to_device_bytes_total\{[^}]*\} ([0-9.eE+]+)"
    )
    for line in open(METRICS):
        m = pat.search(line)
        if not m or m.group(1) != mode or int(m.group(2)) != mbt or m.group(3) != work:
            continue
        vals[m.group(4)] = [float(m.group(i)) for i in range(5, 8)]
    if not vals["before"] or not vals["after"]:
        return None
    b, a = vals["before"], vals["after"]
    return [a[i] - b[i] for i in range(3)]


print("mode\tmbt\tworkload\tttft_ms\tttft_p99_ms\titl_ms\titl_p99_ms\tout_tput\tcompleted\tfailed\tcapacity_tokens\tcapacity_x\thits\tmisses\th2d_bytes")
for mbt in [4096, 8192, 16384, 32768]:
    for mode in ["on", "off"]:
        cap_tokens, cap_x = capacity(mode, mbt)
        for work in WORKLOADS:
            r = result(mode, mbt, work)
            if r is None:
                print(f"{mode}\t{mbt}\t{work}\tMISSING")
                continue
            d = metric_deltas(mode, mbt, work)
            hits, misses, h2d = d if d else ("", "", "")
            print(
                f"{mode}\t{mbt}\t{work}\t{r['ttft']:.2f}\t{r['ttft_p99']:.2f}\t"
                f"{r['itl']:.3f}\t{r['itl_p99']:.3f}\t{r['out_tput']:.4f}\t"
                f"{r['completed']}\t{r['failed']}\t{cap_tokens or ''}\t{cap_x or ''}\t"
                f"{hits}\t{misses}\t{h2d}"
            )
