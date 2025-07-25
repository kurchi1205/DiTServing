import asyncio
import aiohttp
import sys
import os
import json
from datetime import datetime
from PIL import Image
from collections import defaultdict

sys.path.insert(0, "../")
from custom_serve.utils.logger import get_logger
from custom_serve.serving.config_loader import ConfigLoader

logger = get_logger(__name__)

class CachingClient:
    def __init__(self, config_path="../cystom_serve/configs/config.yaml"):
        self.config = ConfigLoader(config_path).config
        self.server_url = self.config["server"]["url"]
        self.poll_interval = self.config["client"]["poll_interval"]

    async def start_background_process(self):
        url = f"{self.server_url}/start_background_process"
        params = {"model": ""}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Background process started: {result}")
                        return result
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to start background process: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error starting background process: {e}")
            return {"error": str(e)}

    async def change_caching_interval(self, interval: int):
        url = f"{self.server_url}/change_caching_interval"
        data = {"cache_interval": interval}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Caching interval changed: {result}")
                        return result
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to change caching interval: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Exception during change_caching_interval: {e}")
            return {"error": str(e)}

    async def add_request(self, prompt, timesteps_left):
        url = f"{self.server_url}/add_request"
        data = {"prompt": prompt, "timesteps_left": timesteps_left}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to add request. Server response: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            return {"error": str(e)}

    async def get_output(self):
        url = f"{self.server_url}/get_output"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("completed_requests", [])
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to fetch outputs. Server response: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error retrieving outputs: {e}")
            return {"error": str(e)}

    async def poll_for_outputs(self, path):
        while True:
            outputs = await self.get_output()
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            if isinstance(outputs, list) and outputs:
                for output in outputs:
                    start_time = datetime.fromisoformat(output["timestamp"])
                    end_time = datetime.fromisoformat(output["time_completed"])
                    time_taken = (end_time - start_time).total_seconds()
                    output["time_taken"] = time_taken
                    print(f"Completed Request: {json.dumps(output, indent=2)}")
                    image_path = output["image"]
                    img = Image.open(image_path)
                    img.save(path)
                break
            await asyncio.sleep(self.poll_interval)

    async def run(self, interval: int, prompt: str, timesteps_left: int, key: str):
        # logger.info("Starting background process...")
        # await self.start_background_process()

        # logger.info(f"Setting caching interval to {interval}")
        # await self.change_caching_interval(interval)

        logger.info("Submitting inference request...")
        await self.add_request(prompt, timesteps_left)

        logger.info("Polling for output...")
        path = f"/home/DiTServing/assets/partiprompts_sd3_custom/{key}_cache_{interval}.png"
        await self.poll_for_outputs(path)


    async def start_bg_process(self):
        logger.info("Starting background process...")
        await self.start_background_process()


async def process_prompts(client, prompts_dict, challenges_dict, interval):
    # Group prompts by challenge
    challenge_prompts = defaultdict(list)
    for key, prompt in prompts_dict.items():
        challenge = challenges_dict.get(f"{key}_challenge")
        if challenge:
            challenge_prompts[challenge].append((key, prompt))

    # Take first 20 per challenge
    filtered_prompts = {}
    for entries in challenge_prompts.values():
        for key, prompt in entries[:20]:
            filtered_prompts[key] = prompt

    for key, prompt in filtered_prompts.items():
        print(f"Processing {key} (cache_{interval})")
        try:
            await client.run(interval, prompt, timesteps_left=50, key=key)
        except Exception as e:
            logger.error(f"Error processing {key}: {e}")



if __name__ == "__main__":
    client = CachingClient()
    interval_list = [5]

    prompt_path = "parti_prompts.json"
    challenge_path = "parti_challenges.json"
    prompts = json.load(open(prompt_path))
    challenges = json.load(open(challenge_path))

    for interval in interval_list:
        asyncio.run(process_prompts(client, prompts, challenges, interval))

