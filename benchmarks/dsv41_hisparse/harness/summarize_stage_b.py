import glob
import json
import os
import re

BASE = "/dockerdata/yeshihai_dsv41_buffer/results/stage_b_20261107_r1"
METRICS = os.path.join(BASE, "metrics.log")
WORKLOADS = [
    "b_131k_c1",
    "b_262k_c1",
    "b_524k_c1",
    "b_1m_c1",
    "b_131k_c8",
    "b_131k_c16",
    "b_131k_c32",
    "b_4k_c32",
    "d_prewarm",
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


def capacity(mode):
    path = os.path.join(BASE, f"capacity_{mode}.txt")
    if not os.path.exists(path):
        return None, None
    text = open(path).read()
    m = re.search(r"([0-9,]+) tokens.*?([0-9.]+)x", text)
    if not m:
        return None, None
    return int(m.group(1).replace(",", "")), float(m.group(2))


def metric_deltas(mode, work):
    if mode != "on" or not os.path.exists(METRICS):
        return None
    vals = {"before": None, "after": None}
    pat = re.compile(
        r"\] (on|off) mbt=4096 (\S+) (before|after) "
        r"vllm:hisparse_cache_hits_total\{[^}]*\} ([0-9.eE+]+) "
        r"vllm:hisparse_cache_misses_total\{[^}]*\} ([0-9.eE+]+) "
        r"vllm:hisparse_host_to_device_bytes_total\{[^}]*\} ([0-9.eE+]+)"
    )
    for line in open(METRICS):
        m = pat.search(line)
        if not m or m.group(1) != mode or m.group(2) != work:
            continue
        vals[m.group(3)] = [float(m.group(i)) for i in range(4, 7)]
    if vals["before"] is None or vals["after"] is None:
        return None
    return [vals["after"][i] - vals["before"][i] for i in range(3)]


def accuracy(mode):
    path = f"/dockerdata/yeshihai_dsv41_buffer/results/dsv41_gsm8k_stage_b_{mode}_4096.json"
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    if isinstance(data, list):
        return sum(1 for x in data if x.get("correct")) / max(len(data), 1)
    return data.get("accuracy")


print(
    "mode\tworkload\tttft_ms\tttft_p99_ms\titl_ms\titl_p99_ms\tout_tput\t"
    "completed\tfailed\tcapacity_tokens\tcapacity_x\thits\tmisses\th2d_bytes\tmiss_rate"
)
for mode in ["on", "off"]:
    cap_tokens, cap_x = capacity(mode)
    for work in WORKLOADS:
        r = result(mode, work)
        if r is None:
            print(f"{mode}\t{work}\tMISSING")
            continue
        d = metric_deltas(mode, work)
        if d:
            hits, misses, h2d = d
            rate = misses / (hits + misses) if hits + misses else 0.0
            rate = f"{rate:.4f}"
        else:
            hits = misses = h2d = ""
            rate = ""
        print(
            f"{mode}\t{work}\t{r['ttft']:.2f}\t{r['ttft_p99']:.2f}\t"
            f"{r['itl']:.3f}\t{r['itl_p99']:.3f}\t{r['out_tput']:.4f}\t"
            f"{r['completed']}\t{r['failed']}\t{cap_tokens or ''}\t{cap_x or ''}\t"
            f"{hits}\t{misses}\t{h2d}\t{rate}"
        )
    print(f"{mode}\tGSM8K_200\taccuracy={accuracy(mode)}")
