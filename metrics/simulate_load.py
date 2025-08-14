import json
import random
import asyncio
import aiohttp
import os
from datetime import datetime
import time

# Config
PROMPT_FILE = "/home/fast-dit-serving/metrics/testing_prompts.json"
completed_log_path = "/home/fast-dit-serving/metrics/completed_requests_log_100_req.json"
SAVE_INTERVAL = 5

# Load prompts
prompts = list(json.load(open(PROMPT_FILE)).items())
accumulated = {"completed_requests": []}


def append_to_json_file(new_requests, file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {"completed_requests": []}
    else:
        existing_data = {"completed_requests": []}

    existing_data["completed_requests"].extend(new_requests)
    print("Dumping to json")
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=2)


async def poll_until_completed(get_url):
    global accumulated
    last_flush_time = time.time()
    FLUSH_INTERVAL_SECONDS = 10  # Flush even if less than SAVE_INTERVAL

    try:
        while True:
            await asyncio.sleep(1)

            async with aiohttp.ClientSession() as session:
                async with session.get(get_url) as response:
                    if response.status == 200:
                        results = await response.json()
                        new_requests = results.get("completed_requests", [])

                        if new_requests:
                            for req in new_requests:
                                req_id = req.get("request_id")
                                existing_ids = {r.get("request_id") for r in accumulated["completed_requests"]}
                                if req_id not in existing_ids:
                                    accumulated["completed_requests"].append(req)
                                    print(f"Logged request {req_id}")

                            # Save if batch size reached
                            if len(accumulated["completed_requests"]) >= SAVE_INTERVAL:
                                append_to_json_file(accumulated["completed_requests"], completed_log_path)
                                accumulated["completed_requests"].clear()

            # Timed flush even if batch not full
            if time.time() - last_flush_time > FLUSH_INTERVAL_SECONDS:
                if accumulated["completed_requests"]:
                    append_to_json_file(accumulated["completed_requests"], completed_log_path)
                    print(f"Timed flush: saved {len(accumulated['completed_requests'])} requests.")
                    accumulated["completed_requests"].clear()
                last_flush_time = time.time()

    except Exception as e:
        print(f"Error while polling: {e}")

    # Final save
    if accumulated["completed_requests"]:
        append_to_json_file(accumulated["completed_requests"], completed_log_path)
        print(f"Final flush: saved {len(accumulated['completed_requests'])} requests.")
        accumulated["completed_requests"].clear()


async def send_request(prompt_key, prompt_text, host="http://localhost:8000"):
    add_url = f"{host}/add_request"
    data = {
        "prompt": prompt_text,
        "timesteps_left": random.choice([50])
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(add_url, json=data) as response:
                if response.status == 200:
                    print(f"Submitted {prompt_key}")
                else:
                    print(f"Error submitting {prompt_key}: {response.status}")
    except Exception as e:
        print(f"Exception adding {prompt_key}: {e}")


async def simulate_load(num_requests=10, delay_between=0.5):
    tasks = []

    for _ in range(num_requests):
        prompt_key, prompt_text = random.choice(prompts)
        tasks.append(send_request(prompt_key, prompt_text))
        await asyncio.sleep(delay_between)

    await asyncio.gather(*tasks)

async def simulate_constant_throughput(rate_per_sec=2, total_duration=30):
    interval = 1.0 / rate_per_sec
    end_time = asyncio.get_event_loop().time() + total_duration
    while asyncio.get_event_loop().time() < end_time:
        prompt_key, prompt_text = random.choice(prompts)
        asyncio.create_task(send_request(prompt_key, prompt_text))
        await asyncio.sleep(interval)


async def main():
    # Start background polling
    get_url = "http://localhost:8000/get_output"
    poll_task = asyncio.create_task(poll_until_completed(get_url))

    # Simulate steady load
    # await simulate_load(num_requests=6, delay_between=1)
    await(simulate_constant_throughput(4, 25))

    # Keep polling for a while to finish all responses
    await asyncio.sleep(200)
    poll_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
