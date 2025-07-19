import argparse
import json

def get_benchmark_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["benchmarks"]

def get_table(benchmark):
    headers = ["Group", "Name", "Min Time", "Max Time", "Mean Time", "Std Dev", "Median", "Rounds"]
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

def print_table(headers, rows, file_path):
    with open(file_path, 'w') as file:
        widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        line_format = ' | '.join(f'{{:<{width}}}' for width in widths)

        print(line_format.format(*headers), file=file)
        print('-' * sum(widths + [3 * (len(headers) - 1)]), file=file)  # account for ' | ' separators

        for row in rows:
            print(line_format.format(*row), file=file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process benchmark data from a JSON file.")
    parser.add_argument('--file', type=str, help="Path to the JSON file containing the benchmark data.")
    parser.add_argument('--output', type=str, help="Path to the output text file where the table will be saved.")

    args = parser.parse_args()
    benchmark = get_benchmark_data(args.file)
    headers, rows = get_table(benchmark)
    print_table(headers, rows, args.output)
