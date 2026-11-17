"""Read only and offline: a queue model for the backlog design. It contacts no service and reads no /var/ossec file: only the repository's sample and the sanitized Phase 2.6E evidence.

It answers "how large would a backlog get at a given drain rate" for the one day available second by second, and summarizes daily volumes from the archive analysis. It uses EVENT time as if
it were arrival time. That's an assumption: the real arrival rate isn't known. So these are model outputs for sizing, and not measurements of the listener.
usage: python -I -B backlog_model.py SAMPLE_ALERTS 2_6E_ARCHIVE_JSON 2_6E_DETAILS_JSON"""
import collections
import json
import sys


def simulate(per_second, drain):
    """Fluid model over the seconds from first to last. Returns (largest backlog, seconds until it empties after the largest backlog, seconds with any backlog)."""
    start, end = min(per_second), max(per_second)
    backlog = peak = peak_at = busy = 0
    empty_after = None
    for second in range(start, end + 1):
        backlog = max(0, backlog + per_second.get(second, 0) - drain)
        busy += backlog > 0
        if backlog > peak:
            peak, peak_at, empty_after = backlog, second, None
        if backlog == 0 and empty_after is None and peak:
            empty_after = second - peak_at
    return peak, empty_after if empty_after is not None else "not within the day", busy


def main():
    sample, archive_path, details_path = sys.argv[1:4]
    per_second = collections.Counter()
    sizes = []
    for raw in open(sample, "rb"):
        sizes.append(len(raw))
        per_second[int(json.loads(raw)["id"].split(".")[0])] += 1
    sizes.sort()
    print(f"sample: {len(sizes)} alerts over {max(per_second) - min(per_second)} s; busiest second {max(per_second.values())}; alert bytes median {sizes[len(sizes) // 2]}, p99 {sizes[int(len(sizes) * .99)]}, max {sizes[-1]}, mean {sum(sizes) // len(sizes)}")
    print("\ndrain/s  largest backlog  seconds to empty after it  seconds with a backlog   (MODEL, sample day, event time taken as arrival time)")
    for drain in (5, 10, 20, 50, 100, 200, 400, 1000):
        peak, empty, busy = simulate(per_second, drain)
        print(f"{drain:>7}  {peak:>15}  {str(empty):>25}  {busy:>22}")

    archive = json.load(open(archive_path))
    days = sorted(((f.get("alerts", 0), f["file"]) for f in archive["files"]), reverse=True)
    counts = sorted(c for c, _ in days)
    print(f"\narchive: {sum(counts)} alerts in {len(counts)} files; median file {counts[len(counts) // 2]}; 90th percentile {counts[int(len(counts) * .9)]}")
    print("largest files, alerts, and that day's mean per second over 86,400 s:")
    for count, name in days[:5]:
        print(f"  {name}: {count}, {count / 86400:.2f}/s")
    details = json.load(open(details_path))
    print(f"busiest event-time second {details['busiest_second_anywhere']}; seconds with 100 or more {details['seconds_with_100_or_more']}; largest id seconds drop {details['largest_id_seconds_drop']}")
    mean = sum(sizes) // len(sizes)
    print(f"\nspool bytes if whole alerts were kept, at the sample's mean of {mean} bytes:")
    for label, count in (("largest day", days[0][0]), ("busiest second", details["busiest_second_anywhere"]), ("archive, nine months", sum(counts))):
        print(f"  {label}: {count} alerts, about {count * mean / 1e6:.1f} MB of alert text, before SQLite's overhead")


main()
