import asyncio
import aiohttp
import json
from datetime import datetime
import sys
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

    async def poll_for_outputs(self):
        while True:
            outputs = await self.get_output()
            if isinstance(outputs, list) and outputs:
                for output in outputs:
                    start_time = datetime.fromisoformat(output["timestamp"])
                    end_time = datetime.fromisoformat(output["time_completed"])
                    time_taken = (end_time - start_time).total_seconds()
                    output["time_taken"] = time_taken
                    print(f"Completed Request: {json.dumps(output, indent=2)}")
                break
            await asyncio.sleep(self.poll_interval)

    async def run(self, interval: int, prompt: str, timesteps_left: int):
        # logger.info("Starting background process...")
        await self.start_background_process()

        # logger.info(f"Setting caching interval to {interval}")
        await self.change_caching_interval(interval)

        logger.info("Submitting inference request...")
        await self.add_request(prompt, timesteps_left)

        logger.info("Polling for output...")
        await self.poll_for_outputs()


    async def start_bg_process(self):
        logger.info("Starting background process...")
        await self.start_background_process()

if __name__ == "__main__":
    client = CachingClient()
    # asyncio.run(client.start_bg_process())
    interval_list = [5]
    for interval in interval_list:
        # interval = 0  # Set desired caching interval
    #     prompt = '''pinkfantasybabes, 
    # Close-up of a woman's face wearing an ornate, intricate masquerade mask that covers her eyes and upper nose...'''
#         prompt = '''stars, water, brilliantly
# gorgeous large scale scene,
# a little girl, in the style of
# dreamy realism, light gold
# and amber, blue and pink,
# brilliantly illuminated in the
# background'''
        prompt = '''colored sketch in the style of ck-ccd, young Asian woman wearing a motorcycle helmet, 
        long loose platinum hair,
         sitting on a large powerful motorcycle, leather jacket, sunset, in orange hues'''
        # prompt = '''Serene Young Woman Portrait with Iridescent Shawl Art'''
        prompt = '''Vintage Portrait of a Young Person with Quill Illustration Art'''
        # prompt = '''Brave Acorn Knight Defending Miniature Art'''
        # prompt = '''Cozy Bedroom Scene with Woman Taking a Mirror Selfie Art'''
        prompt = '''Cheerful Cartoon Character running in Vibrant  Illustration Art'''
        prompt = '''Whimsical Garden Gnome with Flowers and Butterflies Poster'''
        prompt = '''Pirate ship trapped in a
cosmic maelstrom nebula,
rendered in cosmic beach
whirlpool engine,
volumetric lighting,
spectacular, ambient lights,
light pollution, cinematic
atmosphere, art nouveau
style, illustration art artwork
by SenseiJaye, intricate
detail'''

        timesteps_left = 50
        try:
            asyncio.run(client.run(interval, prompt, timesteps_left))
        except KeyboardInterrupt:
            logger.info("Client stopped by user.")
        except Exception as e:
            logger.error(f"Client error: {e}")
