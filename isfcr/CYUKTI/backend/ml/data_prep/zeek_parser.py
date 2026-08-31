import csv

DEFAULT_INPUT_FILE = "../../../data/raw_logs/conn.log"
DEFAULT_OUTPUT_FILE = "../artifacts/zeek_flows.csv"

FIELDNAMES = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
    "proto", "duration", "bytes", "packets",
]


def parse_conn_log(input_file: str = DEFAULT_INPUT_FILE) -> list[dict]:
    flows = []
    with open(input_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            try:
                flows.append({
                    "timestamp": fields[0],
                    "src_ip": fields[2],
                    "src_port": fields[3],
                    "dst_ip": fields[4],
                    "dst_port": fields[5],
                    "proto": fields[6],
                    "duration": fields[8],
                    "bytes": fields[9],
                    "packets": fields[16],
                })
            except IndexError:
                continue

    return flows


def write_flows(flows: list[dict], output_file: str = DEFAULT_OUTPUT_FILE) -> None:
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        for flow in flows:
            writer.writerow(flow)


if __name__ == "__main__":
    flows = parse_conn_log()
    write_flows(flows)
    print("Zeek flows extracted:", len(flows))
