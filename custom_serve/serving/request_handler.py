import sys
import asyncio
import uuid
from datetime import datetime
try:
    from scheduler import Scheduler
    from constants import RequestStatus
except ImportError:
    from .scheduler import Scheduler
    from .constants import RequestStatus

sys.path.append("../")
from utils.logger import get_logger

logger = get_logger(__name__)

class RequestPool:
    def __init__(self):
        self.requests = {}
        self.active_queue = asyncio.Queue()
        self.attn_queue = asyncio.Queue()
        self.output_pool = asyncio.Queue()
        self.lock = asyncio.Lock()
        logger.info("Initialized RequestPool.")

    def add_request_to_pool(self, request):
        """Add a new request to the pool."""
        self.requests[request["request_id"]] = request
        logger.info(f"Request added to pool: {request['request_id']} (Prompt: {request['prompt']})")

    async def add_to_active_queue(self, request_id):
        """Add a request to the active queue."""
        async with self.lock:
            await self.active_queue.put(request_id)
        logger.info(f"Request added to active queue: {request_id}")

    async def add_to_output_pool(self, request):
        """Add a completed request to the output pool."""
        async with self.lock:
            await self.output_pool.put(request)
        logger.info(f"Request added to output pool: {request['request_id']} (Prompt: {request['prompt']})")

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
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.request_pool = RequestPool()
        self.scheduler = Scheduler(batch_size=config.get("batch_size", 1))
        self.cache_interval = config.get("cache_interval", 5)
        self.max_requests = config.get("batch_size", 1) * self.cache_interval
        logger.info(f"Initialized RequestHandler with batch_size={self.scheduler.batch_size}, "
                    f"cache_interval={self.cache_interval}, max_requests={self.max_requests}")

    def create_request(self, prompt, timesteps_left):
        request = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "status": RequestStatus.PENDING,
            "timesteps_left": timesteps_left,
            "cache_interval": 0,  # Default cache interval
            "prompt": prompt
        }
        logger.info(f"Created new request: {request['request_id']} (Prompt: {request['prompt']})")
        return request

    async def add_request(self, prompt, timesteps_left):
        request = self.create_request(prompt, timesteps_left)
        self.request_pool.add_request_to_pool(request)
        
    def update_timesteps_left(self, request_id):
        if request_id in self.request_pool.requests:
            self.request_pool.requests[request_id]["timesteps_left"] -= 1
            logger.debug(f"Updated timesteps_left for request {request_id}: "
                        f"{self.request_pool.requests[request_id]['timesteps_left']}")

    def update_status(self, request_id):
        if request_id in self.request_pool.requests:
            request = self.request_pool.requests[request_id]
            if request["timesteps_left"] == 0:
                request["status"] = RequestStatus.COMPLETED
                logger.info(f"Request {request_id} marked as COMPLETED.")
            else:
                request["status"] = RequestStatus.IN_PROGRESS
                logger.debug(f"Request {request_id} marked as IN_PROGRESS.")

    async def process_request(self, model):
        logger.info("Starting request processing cycle...")
        while True:
            await self.scheduler.add_to_active_request_queue(self.request_pool, self.max_requests)

            # Step 1: Shift requests to attn_queue using the scheduler
            await self.scheduler.shift_to_attn_queue(self.request_pool, self.max_requests)

            # Step 2: Process attention requests in a batch
            attn_requests = await self.request_pool.get_all_attn_requests()
            if attn_requests:
                await self._process_batch(model, attn_requests, requires_attention=True)

            # Step 3: Process non-attention active requests in a batch
            active_requests = await self.request_pool.get_all_active_requests()
            if active_requests:
                await self._process_batch(model, active_requests, requires_attention=False)
            await asyncio.sleep(0.01)  # Avoid high CPU usage

    async def _process_batch(self, model, batch, requires_attention):
        async def process_batch_conc(request_id):
            """
            Process an individual request.
            """
            request = self.request_pool.requests[request_id]

            # Perform attention-specific or general processing
            if requires_attention:
                # Attention-specific processing: recompute latent
                # request["latent"] = model.compute_latent(request["prompt"])
                request["cache_interval"] = 5  # Reset cache interval
                logger.info(f"Recomputed latent for request {request_id}. Cache interval reset.")
            else:
                # General processing: decrement cache interval
                request["cache_interval"] -= 1
                logger.debug(f"Decremented cache interval for request {request_id}: "
                            f"{request['cache_interval']}")

            # Decrement timesteps left
            request["timesteps_left"] -= 1
            logger.debug(f"Decremented timesteps_left for request {request_id}: "
                        f"{request['timesteps_left']}")

            self.update_status(request_id)

            # Add processed request back to the active queue if not completed
            if request["status"] != RequestStatus.COMPLETED:
                await self.request_pool.add_to_active_queue(request_id)
            elif request["status"] == RequestStatus.COMPLETED:
                logger.info(f"Request {request_id} completed. Moving to output pool.")
                await self.request_pool.add_to_output_pool(request)
                del self.request_pool.requests[request_id]

        # Create tasks for all requests in the batch
        tasks = [process_batch_conc(request_id) for request_id in batch]

        # Run all tasks concurrently
        await asyncio.gather(*tasks)
