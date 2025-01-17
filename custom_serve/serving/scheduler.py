import asyncio
try:
    from constants import RequestStatus
except ImportError:
    from .constants import RequestStatus
import sys
sys.path.append("../")
from utils.logger import get_logger

logger = get_logger(__name__)

class Scheduler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.lock = asyncio.Lock()
        logger.info(f"Scheduler initialized with batch_size={self.batch_size}")

    async def add_to_active_request_queue(self, request_pool, max_request_size):
        async with self.lock:
            # Determine how many more requests can be added to the active queue
            available_slots = max_request_size - request_pool.active_queue.qsize()
            if available_slots <= 0:
                logger.warning(f"Active queue is full. Max size: {max_request_size}")
                return

            # Calculate the actual number of requests to add in this batch
            requests_to_add = min(self.batch_size, available_slots)
            added_requests = 0

            # Iterate through pending requests in the request pool
            for request_id, request in request_pool.requests.items():
                if request["status"] == RequestStatus.PENDING:
                    # Add request to the active queue
                    await request_pool.add_to_active_queue(request_id)
                    request["status"] = RequestStatus.IN_PROGRESS
                    logger.debug(f"Request {request_id} added to active queue.")
                    added_requests += 1

                    # Stop if we've added the required number of requests for this batch
                    if added_requests >= requests_to_add:
                        break

            if added_requests == 0:
                logger.debug("No pending requests available to add to the active queue.")
            else:
                logger.info(f"Added {added_requests} requests to the active queue (Batch size: {self.batch_size}).")


    async def shift_to_attn_queue(self, request_pool, max_active_requests):
        """Shift requests requiring attention to the attention queue."""
        async with self.lock:
            active_requests = []

            # Step 1: Process active queue first
            while not request_pool.active_queue.empty():
                request_id = await request_pool.active_queue.get()
                request = request_pool.requests[request_id]

                # Check if the request requires attention
                if request["cache_interval"] == 0:
                    await request_pool.attn_queue.put(request_id)
                    logger.debug(f"Request {request_id} shifted to attention queue.")
                else:
                    active_requests.append(request_id)

            # Re-populate active queue with remaining requests
            for request_id in active_requests:
                await request_pool.active_queue.put(request_id)
            logger.debug(f"Repopulated active queue with {len(active_requests)} requests.")

            # Step 2: Add new requests from the pool if attn_queue is not full
            for request_id, request in request_pool.requests.items():
                if request_pool.attn_queue.qsize() >= self.batch_size:
                    logger.debug("Attention queue is full. Stopping shift to attention queue.")
                    break  # Stop if attn_queue is full
                if request["status"] == RequestStatus.PENDING:
                    if (request_pool.active_queue.qsize() + request_pool.attn_queue.qsize()) < max_active_requests:
                        await request_pool.attn_queue.put(request_id)
                        request["status"] = RequestStatus.IN_PROGRESS
                        logger.debug(f"New request {request_id} added to attention queue.")

    async def shift_to_active_queue_from_attn(self, request_pool):
        """
        Transfer requests from the attention queue back to the active queue
        after attention-specific processing is complete.
        """
        async with self.lock:
            transferred_requests = 0
            while not request_pool.attn_queue.empty():
                request_id = await request_pool.attn_queue.get()
                # Add the request back to the active queue
                await request_pool.add_to_active_queue(request_id)
                logger.debug(f"Request {request_id} shifted back to active queue from attention queue.")
                transferred_requests += 1
            logger.debug(f"Transferred {transferred_requests} requests from attention queue to active queue.")
