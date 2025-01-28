import aiohttp
import asyncio
import time
from datetime import datetime
import statistics
from tabulate import tabulate

class ProcessRequestsBenchmark:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.response_times = []
        self.success_count = 0
        self.error_count = 0
        self.errors = []

    async def benchmark_single_request(self, session):
        """Execute a single process_requests benchmark"""
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/process_requests") as response:
                end_time = time.time()
                response_time = end_time - start_time
                self.response_times.append(response_time)
                
                if response.status == 200:
                    self.success_count += 1
                else:
                    self.error_count += 1
                    self.errors.append(f"Status code: {response.status}")
                
                return {
                    "response_time": response_time,
                    "status": response.status,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.error_count += 1
            self.errors.append(str(e))
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def calculate_statistics(self):
        """Calculate detailed statistics from the benchmark results"""
        if not self.response_times:
            return None

        return {
            "total_requests": len(self.response_times) + self.error_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": (self.success_count / (self.success_count + self.error_count)) * 100,
            "min_response_time": min(self.response_times) if self.response_times else None,
            "max_response_time": max(self.response_times) if self.response_times else None,
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else None,
            "median_response_time": statistics.median(self.response_times) if self.response_times else None,
            "p95_response_time": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else None,
            "p99_response_time": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else None,
            "std_dev": statistics.stdev(self.response_times) if len(self.response_times) > 1 else None
        }

    def save_results(self, stats, filename="process_requests_benchmark.txt"):
        """Save benchmark results to a file in a detailed table format"""
        with open(filename, 'w') as f:
            # Write header
            f.write(f"Process Requests Benchmark Results - {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

            # Write general statistics
            general_stats = [
                ["Total Requests", stats["total_requests"]],
                ["Successful Requests", stats["successful_requests"]],
                ["Failed Requests", stats["failed_requests"]],
                ["Success Rate", f"{stats['success_rate']:.2f}%"]
            ]
            f.write("General Statistics\n")
            f.write(tabulate(general_stats, tablefmt="grid"))
            f.write("\n\n")

            # Write timing statistics
            timing_stats = [
                ["Metric", "Value (seconds)"],
                ["Minimum Response Time", f"{stats['min_response_time']:.4f}"],
                ["Maximum Response Time", f"{stats['max_response_time']:.4f}"],
                ["Average Response Time", f"{stats['avg_response_time']:.4f}"],
                ["Median Response Time", f"{stats['median_response_time']:.4f}"],
                ["95th Percentile", f"{stats['p95_response_time']:.4f}" if stats['p95_response_time'] else "N/A"],
                ["99th Percentile", f"{stats['p99_response_time']:.4f}" if stats['p99_response_time'] else "N/A"],
                ["Standard Deviation", f"{stats['std_dev']:.4f}" if stats['std_dev'] else "N/A"]
            ]
            f.write("Timing Statistics\n")
            f.write(tabulate(timing_stats, headers="firstrow", tablefmt="grid"))
            f.write("\n\n")

            # Write errors if any
            if self.errors:
                f.write("Errors Encountered\n")
                f.write("-" * 40 + "\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"{i}. {error}\n")

async def run_benchmark(num_requests=100, delay_between_requests=0.1):
    """Run the benchmark with specified parameters"""
    benchmark = ProcessRequestsBenchmark()
    
    print(f"Starting benchmark with {num_requests} requests...")
    print(f"Delay between requests: {delay_between_requests} seconds")
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            await benchmark.benchmark_single_request(session)
            if i < num_requests - 1:  # Don't delay after the last request
                await asyncio.sleep(delay_between_requests)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} requests...")
    
    stats = benchmark.calculate_statistics()
    benchmark.save_results(stats)
    print("\nBenchmark completed. Results saved to process_requests_benchmark.txt")

if __name__ == "__main__":
    asyncio.run(run_benchmark())