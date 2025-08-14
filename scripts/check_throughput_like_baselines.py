#!/usr/bin/env python3
"""
Batch test client for SD3 server (sequential, no timeout)
Usage:
  python run_batch.py --json prompts.json [--timesteps 50] [--server-url http://localhost:8000]
"""
import argparse
import asyncio
import aiohttp
import sys
import time
import json
from typing import List, Dict, Any, Tuple, Optional

POLL_INTERVAL_SEC = 1  # seconds between polls


async def start_background(session: aiohttp.ClientSession, server_url: str) -> None:
    print("1) Starting background process...")
    async with session.post(f"{server_url}/start_background_process", params={"model": ""}) as resp:
        if resp.status == 200:
            print("   Background process started")
        else:
            print(f"   Background process response: {resp.status}")


async def submit_request(session: aiohttp.ClientSession, server_url: str, prompt: str, timesteps: int) -> Dict[str, Any]:
    print(f"2) Submitting: {prompt!r} (timesteps={timesteps})")
    data = {"prompt": prompt, "timesteps_left": timesteps}
    async with session.post(f"{server_url}/add_request", json=data) as resp:
        if resp.status != 200:
            error = await resp.text()
            raise RuntimeError(f"Request failed: {error}")
        result = await resp.json()
        print(f"   Submitted -> {result}")
        return result


async def poll_for_completion(
    session: aiohttp.ClientSession,
    server_url: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.time()

    while True:  # loop until success
        async with session.get(f"{server_url}/get_output") as resp:
            if resp.status == 200:
                result = await resp.json()
                completed = result.get("completed_requests", [])
                if completed:
                    if request_id is None:
                        return completed[0]
                    for req in completed:
                        if req.get("request_id") == request_id:
                            return req
            else:
                pass

        elapsed = int(time.time() - start_time)
        await asyncio.sleep(POLL_INTERVAL_SEC)


def load_prompts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        # If it has a "prompts" key, treat that as the container
        if "prompts" in data and isinstance(data["prompts"], list):
            for item in data["prompts"]:
                if isinstance(item, str):
                    normalized.append({"prompt": item})
                elif isinstance(item, dict) and "prompt" in item:
                    normalized.append({"prompt": item["prompt"], "timesteps": item.get("timesteps")})
        else:
            # Otherwise, treat each value as a prompt string
            for key, value in data.items():
                if isinstance(value, str):
                    normalized.append({"prompt": value})
                elif isinstance(value, dict) and "prompt" in value:
                    normalized.append({"prompt": value["prompt"], "timesteps": value.get("timesteps")})

    if not normalized:
        raise ValueError("No prompts found in JSON.")
    normalized_100 = normalized[:100]
    return normalized_100


async def run_batch(json_path: str, server_url: str, default_timesteps: int) -> Tuple[int, float, int, int]:
    prompts = load_prompts(json_path)
    total = len(prompts)
    successes = 0
    failures = 0
    per_prompt_times: List[float] = []

    print(f"🚀 Running batch with {total} prompt(s)")
    print(f"Server: {server_url}")
    print("-" * 60)

    batch_start = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            await start_background(session, server_url)

            for i, item in enumerate(prompts, start=1):
                prompt = item["prompt"].strip()
                if not prompt:
                    print(f"[{i}/{total}] Skipping empty prompt")
                    failures += 1
                    continue

                timesteps = int(item.get("timesteps", default_timesteps))

                t0 = time.time()
                try:
                    submit_res = await submit_request(session, server_url, prompt, timesteps)
                    req_id = submit_res.get("request_id")
                    completed = await poll_for_completion(session, server_url, request_id=req_id)

                    dt = time.time() - t0
                    per_prompt_times.append(dt)
                    successes += 1
                except Exception as e:
                    dt = time.time() - t0
                    per_prompt_times.append(dt)
                    failures += 1
                    print(f"   ❌ Failed after {dt:.2f}s: {e}")

                print("-" * 60)
    except aiohttp.ClientConnectorError:
        print(f"Cannot connect to {server_url}")
        print("   Make sure the server is running: python server.py")
        return 0, 0.0, 0, total
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    total_wall = time.time() - batch_start
    return total, total_wall, successes, failures


def main():
    parser = argparse.ArgumentParser(description="Sequential batch test client for SD3 server (no timeout)")
    parser.add_argument("--json", required=True, help="Path to JSON containing prompts.")
    parser.add_argument("--timesteps", "-t", type=int, default=50, help="Default timesteps (overridden in JSON).")
    parser.add_argument("--server-url", "-s", default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")

    args = parser.parse_args()

    if args.timesteps <= 0:
        print("Error: --timesteps must be positive")
        sys.exit(1)

    total, total_wall, successes, failures = asyncio.run(run_batch(args.json, args.server_url, args.timesteps))

    if total > 0:
        avg = total_wall / total
        print("\n====== Batch Summary ======")
        print(f"Total prompts       : {total}")
        print(f"Successes / Failures: {successes} / {failures}")
        print(f"Total wall time     : {total_wall:.2f}s")
        print(f"Avg time / prompt   : {avg:.2f}s")
        print("===========================\n")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
