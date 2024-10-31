
# Create a header based on the benchmark dictionary keys
import argparse
import json

def get_benchmark_data(args):
    data = json.load(open(args.file))
    benchmark = data["benchmarks"]
    return benchmark


def get_table(benchmark):
    headers = ["Group", "Name", "Min Time", "Max Time", "Mean Time", "Std Dev", "Median", "Rounds"]
    # Create rows based on data
    rows = []
    for bmk in benchmark:
        stats = bmk["stats"]
        row = [
            bmk["group"],
            bmk["name"],
            round(stats["min"], 3),
            round(stats["max"], 3),
            round(stats["mean"], 3),
            round(stats["stddev"], 3),
            round(stats["median"], 3),
            stats["rounds"]
        ]
        rows.append(row)
    return headers, rows


def print_table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    line_format = ' | '.join(f'{{:<{width}}}' for width in widths)

    print(line_format.format(*headers))
    print('-' * sum(widths + [3 * (len(headers) - 1)]))  # account for ' | ' separators

    for row in rows:
        print(line_format.format(*row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process benchmark data from a JSON file.")
    parser.add_argument('--file', type=str, help="Path to the JSON file containing the benchmark data.")
    
    args = parser.parse_args()
    benchmark = get_benchmark_data(args)
    headers, rows = get_table(benchmark)
    print_table(headers, rows)
