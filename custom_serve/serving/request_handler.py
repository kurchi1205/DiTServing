import sys
import asyncio
import uuid
import time
import torch
from datetime import datetime, timedelta
try:
    from scheduler import Scheduler
    from constants import RequestStatus
    from processor import process_each_timestep
except ImportError:
    from .scheduler import Scheduler
    from .constants import RequestStatus
    from .processor import process_each_timestep, process_each_timestep_batched

sys.path.append("../")
from utils.logger import get_logger
from pipeline.sd3 import CFGDenoiser, SD3LatentFormat

logger = get_logger(__name__)

class RequestPool:
    def __init__(self, inference_handler):
        self.requests = {}
        self.active_queue = asyncio.Queue()
        self.attn_queue = asyncio.Queue()
        self.output_pool = asyncio.Queue()
        self.lock = asyncio.Lock()
        seed = torch.randint(0, 100000, (1,)).item()
        self.empty_latent = inference_handler.get_empty_latent(1, 1024, 1024, seed, device="cuda")
        self.neg_cond = inference_handler.fix_cond(inference_handler.get_cond(""))
        logger.info("Initialized RequestPool.")

    def add_request_to_pool(self, request):
        """Add a new request to the pool."""
        self.requests[request["request_id"]] = request
        logger.info(f"Request added to pool: {request['request_id']} (Prompt: {request['prompt']})")

    async def check_pending_timeouts(self, pending_timeout, current_time):
        """Check pending requests and mark those exceeding the timeout as failed."""
        async with self.lock:
            for request_id, request in list(self.requests.items()):
                if request["status"] == RequestStatus.PENDING:
                    request_time = datetime.fromisoformat(request["timestamp"])
                    if current_time - request_time > timedelta(seconds=pending_timeout):
                        request["status"] = RequestStatus.FAILED
                        await self.add_to_output_pool(request)
                        logger.info(f"Request {request_id} marked as FAILED due to timeout.")

            completed_requests = []
            while not self.output_pool.empty():
                completed_request = await self.output_pool.get()
                request_time = datetime.fromisoformat(completed_request["completed_timestamp"])
                if current_time - request_time > timedelta(seconds=pending_timeout):
                    logger.info(f"Completed request {completed_request['request_id']} removed after timeout.")
                else:
                    completed_requests.append(completed_request)

            for request in completed_requests:
                await self.output_pool.put(request)           

    async def add_to_active_queue(self, request_id):
        """Add a request to the active queue."""
        async with self.lock:
            await self.active_queue.put(request_id)
        logger.debug(f"Request added to active queue: {request_id}")


    async def add_to_output_pool(self, request):
        """Add a completed request to the output pool."""
        async with self.lock:
            request['completed_timestamp'] = datetime.now().isoformat()
            await self.output_pool.put(request)
        logger.info(f"Request added to output pool: {request['request_id']} (Prompt: {request['prompt']})")
        if request['request_id'] in self.requests:
            del self.requests[request['request_id']]
            logger.debug(f"Request removed from active pool: {request['request_id']}")


    async def get_all_active_requests(self):
        """Fetch all non-attention active requests."""
        requests = []
        async with self.lock:
            while not self.active_queue.empty():
                requests.append(await self.active_queue.get())
        logger.debug(f"Fetched {len(requests)} active requests.")
        return requests

    async def get_all_attn_requests(self):
        """Fetch all attention requests."""
        requests = []
        async with self.lock:
            while not self.attn_queue.empty():
                requests.append(await self.attn_queue.get())
        logger.debug(f"Fetched {len(requests)} attention requests.")
        return requests


class RequestHandler:
    def __init__(self, config=None, inference_handler=None):
        if config is None:
            config = {}
        sys_config = config["system"]
        self.request_pool = RequestPool(inference_handler)
        self.scheduler = Scheduler(batch_size=sys_config.get("batch_size", 1))
        self.cache_interval = sys_config.get("cache_interval", 5)
        self.max_requests = max(sys_config.get("batch_size", 1) * self.cache_interval, 1)
        logger.info(f"Initialized RequestHandler with batch_size={self.scheduler.batch_size}, "
                    f"cache_interval={self.cache_interval}, max_requests={self.max_requests}")
        self.pending_timeout_check = sys_config.get("pending_timeout_check", 200)

    def create_request(self, prompt, timesteps_left):
        request = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "status": RequestStatus.PENDING,
            "current_timestep": 0,
            "timesteps_left": timesteps_left,
            "cache_interval": 0,  # Default cache interval
            "prompt": prompt,
            "cfg_scale": 5.0,
            "context_latent": {},
            "x_latent": {}
        }
        logger.info(f"Created new request: {request['request_id']} (Prompt: {request['prompt']})")
        return request

    async def add_request(self, prompt, timesteps_left):
        request = self.create_request(prompt, timesteps_left)
        self.request_pool.add_request_to_pool(request)
        
    def update_timesteps_left(self, request_id):
        if request_id in self.request_pool.requests:
            self.request_pool.requests[request_id]["timesteps_left"] -= 1
            self.request_pool.requests[request_id]["current_timestep"] += 1
            timesteps_left = self.request_pool.requests[request_id]["timesteps_left"]
            logger.info(f"Updated timesteps_left for request {request_id}: {timesteps_left}")

    def update_status(self, request):
        if request["timesteps_left"] == 0:
            request["status"] = RequestStatus.COMPLETED
            logger.info(f"Request {request['request_id']} marked as COMPLETED.")
        else:
            request["status"] = RequestStatus.IN_PROGRESS
            logger.debug(f"Request {request['request_id']} marked as IN_PROGRESS.")

    async def process_request(self, inference_handler):
        logger.info("Starting request processing cycle...")
        inference_handler.denoiser = CFGDenoiser
        last_timeout_check = datetime.now()
        while True:
            await self.scheduler.add_to_active_request_queue(self.request_pool, self.max_requests)

            # Step 1: Shift requests to attn_queue using the scheduler
            await self.scheduler.shift_to_attn_queue(self.request_pool, self.max_requests)

            # Step 2: Process attention requests in a batch
            attn_requests = await self.request_pool.get_all_attn_requests()
            active_requests = await self.request_pool.get_all_active_requests()
            tasks = []
            if attn_requests:
                tasks.append(self._process_batch(inference_handler, attn_requests, requires_attention=True))
            if active_requests:
                tasks.append(self._process_batch(inference_handler, active_requests, requires_attention=False))
                
            if tasks:
                await asyncio.gather(*tasks)
            # if (datetime.now() - last_timeout_check).total_seconds() > self.pending_timeout_check:
            #     await self.request_pool.check_pending_timeouts(self.pending_timeout_check, datetime.now())
            #     last_timeout_check = datetime.now()
            # if attn_requests:
            #     await self._process_batch(inference_handler, attn_requests, requires_attention=True)

            # # Step 3: Process non-attention active requests in a batch
            # if active_requests:
            #     await self._process_batch(inference_handler, active_requests, requires_attention=False)
            await asyncio.sleep(0.001)       

    def process_batch(self, inference_handler, request_ids, requires_attention):
        # if requires_attention:
        #     process_each_timestep_batched(inference_handler, request_ids, self.request_pool, compute_attention=True)
        # else:
        requests = process_each_timestep_batched(inference_handler, request_ids, self.request_pool, compute_attention=False)
        return requests
        


    def process_batch_seq(self, inference_handler, request_id, requires_attention):
        """
        Process an individual request.
        """
        request = self.request_pool.requests[request_id]
        logger.debug(f"Processing request {request_id} (Prompt: {request['prompt']})")
        # await asyncio.sleep(0.09)
        # Perform attention-specific or general processing]
        # if requires_attention:
            # Attention-specific processing: recompute latent
            # request["latent"] = model.compute_latent(request["prompt"])
        process_each_timestep(inference_handler, request_id, self.request_pool, compute_attention=True)
            # request["cache_interval"] = self.cache_interval  # Reset cache interval
        logger.debug(f"Recomputed latent for request {request_id}. Cache interval reset.")
        # else:
        #     # General processing: decrement cache interval
        #     process_each_timestep(inference_handler, request_id, self.request_pool, compute_attention=False)
        #     # request["cache_interval"] -= 1
        #     logger.debug(f"Decremented cache interval for request {request_id}: "
        #                 f"{request['cache_interval']}")

        # Decrement timesteps left
        # request["timesteps_left"] -= 1
        # request["current_timestep"] += 1
        # if request["timesteps_left"] == 0:
        #     latent = SD3LatentFormat().process_out(request["noise_scaled"])
        #     image = inference_handler.vae_decode(latent)
        #     request["image"] = image
        #     del request["noise_scaled"]
        #     del request["sigmas"]
        #     del request["conditioning"]
        #     del request["neg_cond"]
        #     del request["old_denoised"]
        #     del request["attention"]

        # logger.debug(f"Decremented timesteps_left for request {request_id}: "
        #             f"{request['timesteps_left']}")
        # self.update_status(request)
        # self.request_pool.requests[request_id] = request
        return request


    async def process_batch_conc(self, inference_handler, request, requires_attention):
        # Add processed request back to the active queue if not completed
        # print("Scaled noise: ", request["noise_scaled"].size())
        # print("old_denoised: ", request["old_denoised"].size())
        if requires_attention:
            request["cache_interval"] = self.cache_interval
        else:
            request["cache_interval"] -= 1
        request_id = request["request_id"]
        request["timesteps_left"] -= 1
        request["current_timestep"] += 1
        if request["timesteps_left"] == 0:
            latent = SD3LatentFormat().process_out(request["noise_scaled"])
            image = inference_handler.vae_decode(latent)
            request["image"] = image
            del request["noise_scaled"]
            del request["sigmas"]
            del request["conditioning"]
            del request["neg_cond"]
            del request["old_denoised"]
            del request["context_latent"]
            del request["x_latent"]

        logger.debug(f"Decremented timesteps_left for request {request_id}: "
                    f"{request['timesteps_left']}")
        self.update_status(request)

        if request["status"] != RequestStatus.COMPLETED:
            await self.request_pool.add_to_active_queue(request_id)
        elif request["status"] == RequestStatus.COMPLETED:
            logger.debug(f"Request {request_id} completed. Moving to output pool.")
            await self.request_pool.add_to_output_pool(request)


    async def _process_batch(self, inference_handler, batch, requires_attention):
        # Create tasks for all requests in the batch
        # print("Batch: ", batch)
        # print("requires_attention: ", requires_attention)
        requests = []
        if requires_attention:
            for request_id in batch:
                requests.append(self.process_batch_seq(inference_handler, request_id, requires_attention))
        else:
            requests = self.process_batch(inference_handler, batch, requires_attention)
            # requests.append(self.request_pool.requests[request_id] for request_id in batch)
        tasks = [self.process_batch_conc(inference_handler, request, requires_attention) for request in requests]
        await asyncio.gather(*tasks)   

