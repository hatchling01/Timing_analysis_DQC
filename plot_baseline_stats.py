#!/usr/bin/env python3

import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_stat(stats_obj, key, default=0):
    return stats_obj.get(key, default)


def plot_total_local_cross(stats_list):
    labels = [s["trace_type"] for s in stats_list]
    total_vals = [s["total_events"] for s in stats_list]
    local_vals = [s["local_events"] for s in stats_list]
    cross_vals = [s["cross_module_events"] for s in stats_list]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, local_vals, label="Local events")
    plt.bar(x, cross_vals, bottom=local_vals, label="Cross-module events")
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Event count")
    plt.title("Local vs Cross-Module Events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_local_vs_cross.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(x, total_vals)
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Total event count")
    plt.title("Total Events by Execution Mode")
    plt.tight_layout()
    plt.savefig("plot_total_events.png", dpi=300)
    plt.show()


def plot_hub_requests(stats_list):
    labels = [s["trace_type"] for s in stats_list]
    completed_vals = [s["completed_hub_requests"] for s in stats_list]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, completed_vals)
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Completed hub requests")
    plt.title("Hub Usage by Execution Mode")
    plt.tight_layout()
    plt.savefig("plot_hub_requests.png", dpi=300)
    plt.show()


def plot_per_module_local(stats_list):
    module_ids = ["module_0", "module_1", "module_2", "module_3", "module_4"]

    for s in stats_list:
        vals = [s["per_module_local_events"].get(m, 0) for m in module_ids]

        plt.figure(figsize=(8, 5))
        plt.bar(module_ids, vals)
        plt.ylabel("Local event count")
        plt.title(f"Per-Module Local Events: {s['trace_type']}")
        plt.tight_layout()
        out_name = f"plot_per_module_{s['trace_type']}.png"
        plt.savefig(out_name, dpi=300)
        plt.show()


def plot_sequential_stage_profile(seq_stats):
    stage_counts = seq_stats.get("stage_counts", {})
    stage_cross_counts = seq_stats.get("stage_cross_counts", {})

    if not stage_counts:
        print("No stage data found for sequential trace.")
        return

    stages = sorted(int(k) for k in stage_counts.keys())

    total_vals = []
    for s in stages:
        if str(s) in stage_counts:
            total_vals.append(stage_counts[str(s)])
        else:
            total_vals.append(stage_counts.get(s, 0))

    cross_vals = []
    for s in stages:
        if str(s) in stage_cross_counts:
            cross_vals.append(stage_cross_counts[str(s)])
        else:
            cross_vals.append(stage_cross_counts.get(s, 0))

    plt.figure(figsize=(10, 5))
    plt.bar(stages, total_vals, label="Total events in stage")
    plt.bar(stages, cross_vals, label="Cross-module events in stage")
    plt.xlabel("Stage ID")
    plt.ylabel("Count")
    plt.title("Sequential Modular Stage Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_sequential_stage_profile.png", dpi=300)
    plt.show()


# ============================================================
# New hub-performance plots
# ============================================================

def plot_avg_waiting_time(stats_list):
    labels = [s["trace_type"] for s in stats_list]
    vals = [get_stat(s, "avg_waiting_time_ns", 0) for s in stats_list]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, vals)
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Average waiting time (ns)")
    plt.title("Average Hub Waiting Time by Execution Mode")
    plt.tight_layout()
    plt.savefig("plot_avg_waiting_time.png", dpi=300)
    plt.show()


def plot_avg_turnaround_time(stats_list):
    labels = [s["trace_type"] for s in stats_list]
    vals = [get_stat(s, "avg_turnaround_time_ns", 0) for s in stats_list]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, vals)
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Average turnaround time (ns)")
    plt.title("Average Hub Turnaround Time by Execution Mode")
    plt.tight_layout()
    plt.savefig("plot_avg_turnaround_time.png", dpi=300)
    plt.show()


def plot_max_waiting_time(stats_list):
    labels = [s["trace_type"] for s in stats_list]
    vals = [get_stat(s, "max_waiting_time_ns", 0) for s in stats_list]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, vals)
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Maximum waiting time (ns)")
    plt.title("Maximum Hub Waiting Time by Execution Mode")
    plt.tight_layout()
    plt.savefig("plot_max_waiting_time.png", dpi=300)
    plt.show()


def plot_hub_makespan(stats_list):
    labels = [s["trace_type"] for s in stats_list]
    vals = [get_stat(s, "hub_makespan_ns", get_stat(s, "hub_current_time_ns", 0)) for s in stats_list]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, vals)
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Hub makespan (ns)")
    plt.title("Hub Makespan by Execution Mode")
    plt.tight_layout()
    plt.savefig("plot_hub_makespan.png", dpi=300)
    plt.show()


def plot_nonzero_wait_requests(stats_list):
    labels = [s["trace_type"] for s in stats_list]
    vals = [get_stat(s, "num_waited_requests", 0) for s in stats_list]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, vals)
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Requests with nonzero wait")
    plt.title("Queued Hub Requests by Execution Mode")
    plt.tight_layout()
    plt.savefig("plot_nonzero_wait_requests.png", dpi=300)
    plt.show()


def plot_sequential_stage_wait_profile(seq_stats):
    stage_wait_avg = seq_stats.get("stage_avg_waiting_time_ns", {})
    if not stage_wait_avg:
        print("No per-stage waiting-time data found for sequential trace.")
        return

    stages = sorted(int(k) for k in stage_wait_avg.keys())
    vals = []
    for s in stages:
        if str(s) in stage_wait_avg:
            vals.append(stage_wait_avg[str(s)])
        else:
            vals.append(stage_wait_avg.get(s, 0))

    plt.figure(figsize=(10, 5))
    plt.bar(stages, vals)
    plt.xlabel("Stage ID")
    plt.ylabel("Average waiting time (ns)")
    plt.title("Sequential Modular Stage Waiting-Time Profile")
    plt.tight_layout()
    plt.savefig("plot_sequential_stage_wait_profile.png", dpi=300)
    plt.show()


def print_missing_metric_warnings(stats_list):
    required_new_keys = [
        "avg_waiting_time_ns",
        "avg_turnaround_time_ns",
        "max_waiting_time_ns",
        "hub_makespan_ns",
        "num_waited_requests",
    ]

    for s in stats_list:
        missing = [k for k in required_new_keys if k not in s]
        if missing:
            print(
                f"Warning: {s['trace_type']} stats JSON is missing new metrics: {missing}. "
                "The new plots will show zeros until those fields are exported."
            )


def main():
    monolithic = load_json("monolithic_stats.json")
    static_dist = load_json("static_distributed_stats.json")
    sequential = load_json("sequential_modular_stats.json")

    stats_list = [monolithic, static_dist, sequential]

    print_missing_metric_warnings(stats_list)

    # Existing plots
    plot_total_local_cross(stats_list)
    plot_hub_requests(stats_list)
    plot_per_module_local(stats_list)
    plot_sequential_stage_profile(sequential)

    # New performance plots
    plot_avg_waiting_time(stats_list)
    plot_avg_turnaround_time(stats_list)
    plot_max_waiting_time(stats_list)
    plot_hub_makespan(stats_list)
    plot_nonzero_wait_requests(stats_list)
    plot_sequential_stage_wait_profile(sequential)


if __name__ == "__main__":
    main()
