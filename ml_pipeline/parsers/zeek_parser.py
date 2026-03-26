import csv

INPUT_FILE = "../../data/raw_logs/conn.log"
OUTPUT_FILE = "../dataset/zeek_flows.csv"

flows = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue

        fields = line.strip().split("\t")

        try:
            flow = {
                "timestamp": fields[0],
                "src_ip": fields[2],
                "src_port": fields[3],
                "dst_ip": fields[4],
                "dst_port": fields[5],
                "proto": fields[6],
                "duration": fields[8],
                "bytes": fields[9],
                "packets": fields[16]
            }

            flows.append(flow)

        except:
            continue


with open(OUTPUT_FILE, "w", newline="") as csvfile:

    fieldnames = [
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "proto",
        "duration",
        "bytes",
        "packets"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for f in flows:
        writer.writerow(f)

print("Zeek flows extracted:", len(flows))
