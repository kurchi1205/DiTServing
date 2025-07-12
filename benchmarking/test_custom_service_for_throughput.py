import asyncio
import time
from datetime import datetime
import aiohttp
import numpy as np
import pandas as pd
from tabulate import tabulate
import json

class ThroughputTester:
    def __init__(self):
        self.server_url = "http://localhost:8000"
        self.total_requests = 50
        self.spawn_rate = 10
        self.spawn_interval = 10  # seconds
        self.output_timeout = 120  # seconds
        self.collected_results = []
        self.test_prompts = [
            "A serene mountain landscape at sunset",
            "A futuristic cityscape at night",
            "An abstract painting with vibrant colors",
            "A photorealistic portrait of a cat",
            "A magical forest with glowing mushrooms"
        ]

    async def add_request(self, prompt, timesteps_left):
        url = f"{self.server_url}/add_request"
        data = {"prompt": prompt, "timesteps_left": timesteps_left}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_message = await response.text()
                    print(f"Failed to add request: {error_message}")
                    return None

    async def get_output(self):
        url = f"{self.server_url}/get_output"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("completed_requests", [])
                else:
                    error_message = await response.text()
                    print(f"Failed to get output: {error_message}")
                    return []

    async def start_background_process(self):
        url = f"{self.server_url}/start_background_process"
        params = {"model": ""}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    print("Successfully started background process")
                    return await response.json()
                else:
                    error_message = await response.text()
                    print(f"Failed to start background process: {error_message}")
                    raise Exception(f"Failed to start background process: {error_message}")

    async def monitor_output_pool(self, expected_requests):
        print("\nMonitoring output pool...")
        start_time = time.time()
        last_result_time = start_time
        
        while len(self.collected_results) < expected_requests:
            current_time = time.time()
            
            # Check timeout condition
            if current_time - last_result_time > self.output_timeout:
                print(f"\nTimeout reached! No new results for {self.output_timeout} seconds")
                break
                
            # Get new results
            results = await self.get_output()
            if results:
                self.collected_results.extend(results)
                last_result_time = current_time
                print(f"Collected {len(self.collected_results)}/{expected_requests} results")
            
            await asyncio.sleep(1)

    def calculate_metrics(self):
        if not self.collected_results:
            return None
            
        # Calculate processing times
        processing_times = []
        total_successful = 0
        total_failed = 0
        
        for result in self.collected_results:
            start_time = datetime.fromisoformat(result['timestamp'])
            end_time = datetime.fromisoformat(result['time_completed'])
            processing_time = (end_time - start_time).total_seconds()
            processing_times.append(processing_time)
            
            # Count successes and failures
            if result['status'].upper() == 'COMPLETED':
                total_successful += 1
            elif result['status'].upper() == 'FAILED':
                total_failed += 1
        
        # Calculate statistics
        mean_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate median
        sorted_times = sorted(processing_times)
        n = len(sorted_times)
        mid = n // 2
        median_time = sorted_times[mid] if n % 2 else (sorted_times[mid-1] + sorted_times[mid]) / 2
        
        # Calculate total duration and throughput
        total_time = sum(processing_times)
        requests_per_second = total_successful / sum(processing_times)
        
        metrics = {
            'Total Requests Sent': self.total_requests,
            'Successful Requests': total_successful,
            'Failed Requests': total_failed,
            'Mean Processing Time': f"{mean_time:.2f}s",
            'Median Processing Time': f"{median_time:.2f}s",
            'Overall Throughput': f"{requests_per_second:.2f} req/s",
            'Total Test Duration': f"{total_time:.2f}s"
        }
        
        return metrics

    def print_results(self, metrics):
        # Create a formatted table
        table = [[key, value] for key, value in metrics.items()]
        print("\nThroughput Test Results:")
        print(tabulate(table, headers=['Metric', 'Value'], tablefmt='grid'))
        
        # Save detailed results to JSON
        with open('throughput_test_results.json', 'w') as f:
            json.dump({
                'metrics': metrics,
                'detailed_results': self.collected_results
            }, f, indent=4)

    async def spawn_requests(self):
        print(f"Starting to spawn requests...")
        requests_sent = 0
        
        while requests_sent < self.total_requests:
            batch_size = min(self.spawn_rate, self.total_requests - requests_sent)
            print(f"\nSending batch of {batch_size} requests...")
            
            # Create batch of requests
            tasks = []
            for _ in range(batch_size):
                prompt = np.random.choice(self.test_prompts)
                tasks.append(self.add_request(prompt, timesteps_left=30))
            
            # Send batch
            await asyncio.gather(*tasks)
            requests_sent += batch_size
            print(f"Total requests sent: {requests_sent}/{self.total_requests}")
            
            if requests_sent < self.total_requests:
                print(f"Waiting {self.spawn_interval} seconds before next batch...")
                await asyncio.sleep(self.spawn_interval)

    async def run_test(self):
        await self.start_background_process()

        print(f"Starting throughput test with {self.total_requests} total requests")
        print(f"Spawning {self.spawn_rate} requests every {self.spawn_interval} seconds")
        
        self.start_time = time.time()
        
        # Run spawning and monitoring concurrently
        await asyncio.gather(
            self.spawn_requests(),
            self.monitor_output_pool(self.total_requests)
        )
        
        # Calculate and print metrics
        metrics = self.calculate_metrics()
        if metrics:
            self.print_results(metrics)
        else:
            print("No results collected!")

async def main():
    tester = ThroughputTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())