
import asyncio
from sre_constants import IN
import uuid
from datetime import datetime
from scheduler import Scheduler

class RequestStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"

class RequestPool:
    def __init__(self):
        self.requests = {}
        self.active_queue = asyncio.Queue()
        self.attn_queue = asyncio.Queue()
        self.output_pool = asyncio.Queue()
        self.lock = asyncio.Lock()

    def add_request_to_pool(self, request):
        """Add a new request to the pool."""
        self.requests[request["request_id"]] = request

    async def add_to_active_queue(self, request_id):
        """Add a request to the active queue."""
        async with self.lock:
            await self.active_queue.put(request_id)

    async def add_to_output_pool(self, request):
        """Add a completed request to the output pool."""
        async with self.lock:
            await self.output_pool.put(request)

    async def get_all_active_requests(self):
        """Fetch all non-attention active requests."""
        requests = []
        async with self.lock:
            while not self.active_queue.empty():
                requests.append(await self.active_queue.get())
        return requests

    async def get_all_attn_requests(self):
        """Fetch all attention requests."""
        requests = []
        async with self.lock:
            while not self.attn_queue.empty():
                requests.append(await self.attn_queue.get())
        return requests



class RequestHandler:
    def __init__(self):
        self.request_pool = RequestPool()
        self.scheduler = Scheduler(batch_size=1)

    def create_request(self, prompt, timesteps_left):
        request = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "status": RequestStatus.PENDING,
            "timesteps_left": timesteps_left,
            "cache_interval": 0,  # Default cache interval
            "prompt": prompt
        }
        return request

    async def add_request(self, prompt, timesteps_left):
        request = self.create_request(prompt, timesteps_left)
        self.request_pool.add_request_to_pool(request)
        await self.scheduler.add_to_active_request_queue(self.request_pool, request["request_id"])

    

    def update_timesteps_left(self, request_id):
        if request_id in self.request_pool.requests:
            self.request_pool.requests[request_id]["timesteps_left"] -= 1

    def update_status(self, request_id):
        if request_id in self.request_pool.requests:
            request = self.request_pool.requests[request_id]
            if request["timesteps_left"] == 0:
                request["status"] = RequestStatus.COMPLETED
            else:
                request["status"] = RequestStatus.IN_PROGRESS

    async def process_request(self, model):
        while True:
            # Step 1: Shift requests to attn_queue using the scheduler
            await self.scheduler.shift_to_attn_queue(self.request_pool)

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
        for request_id in batch:
            request = self.request_pool.requests[request_id]

            # Perform attention-specific or general processing
            if requires_attention:
                # Attention-specific processing: recompute latent
                # request["latent"] = model.compute_latent(request["prompt"])
                request["cache_interval"] = 5  # Reset cache interval
            else:
                # General processing: decrement cache interval
                request["cache_interval"] -= 1

            # Decrement timesteps left
            request["timesteps_left"] -= 1

            self.update_status(request_id)

            # Add processed request back to the active queue if not completed
            if request["status"] != RequestStatus.COMPLETED:
                await self.request_pool.add_to_active_queue(request_id)
            elif request["status"] == RequestStatus.COMPLETED:
                print(request["request_id"], request["prompt"], request["status"])
                await self.request_pool.add_to_output_pool(request)
                del self.request_pool.requests[request_id]




    