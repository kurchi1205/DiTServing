import asyncio
from datetime import datetime
from tracemalloc import start
import aiohttp
import json
from utils.logger import get_logger
from serving.config_loader import ConfigLoader

logger = get_logger(__name__)

class Client:
    def __init__(self, config_path="../configs/config.yaml"):
        self.config = ConfigLoader(config_path).config
        self.server_url = self.config["server"]["url"]
        self.poll_interval = self.config["client"]["poll_interval"]

    async def add_request(self, prompt, timesteps_left):
        url = f"{self.server_url}/add_request"
        data = {"prompt": prompt, "timesteps_left": timesteps_left}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        # logger.info(f"Request submitted: {result}")
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
                        # logger.info(f"Retrieved completed requests: {result}")
                        return result.get("completed_requests", [])
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to fetch outputs. Server response: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error retrieving outputs: {e}")
            return {"error": str(e)}

    async def submit_requests(self, requests):
        tasks = [self.add_request(req["prompt"], req["timesteps_left"]) for req in requests]
        results = await asyncio.gather(*tasks)
        return results

    async def poll_for_outputs(self):
        """
        Poll the server for completed requests at regular intervals.
        """
        while True:
            outputs = await self.get_output()
            if isinstance(outputs, list) and outputs:
                for output in outputs:
                    start_time = datetime.fromisoformat(output["timestamp"])
                    end_time = datetime.fromisoformat(output["time_completed"])
                    time_taken = (end_time - start_time).total_seconds()
                    output["time_taken"] = time_taken
                    print(f"Completed Request: {json.dumps(output, indent=2)}")
            await asyncio.sleep(self.poll_interval)


    async def start_background_process(self):
        """
        Trigger the background process on the server.
        """
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
                        logger.error(f"Failed to start background process. Server response: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error starting background process: {e}")
            return {"error": str(e)}
        

    async def run(self):
        await self.start_background_process()
    
        # Example prompts
        requests = [
            # {"prompt": "Generate image of a cat", "timesteps_left": 30},
            # {"prompt": "Generate image of a dog", "timesteps_left": 30},
            {"prompt": "Generate image of a fish", "timesteps_left": 30},
            # {"prompt": "Generate image of a car", "timesteps_left": 30},
        ]

        logger.info("Starting request submission.")
        await self.submit_requests(requests)

        logger.info("Starting polling for outputs.")
        await self.poll_for_outputs()


if __name__ == "__main__":
    client = Client()

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Client stopped by user.")
    except Exception as e:
        logger.error(f"Error running client: {e}")
