import asyncio
from constants import RequestStatus

class Scheduler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.lock = asyncio.Lock()

    async def add_to_active_request_queue(self, request_pool, request_id):
        async with self.lock: 
            await request_pool.add_to_active_queue(request_id)

    async def shift_to_attn_queue(self, request_pool, max_active_requests):
        
        async with self.lock:
            active_requests = []
            
            # Step 1: Process active queue first
            while not request_pool.active_queue.empty():
                request_id = await request_pool.active_queue.get()
                request = request_pool.requests[request_id]

                # Check if the request requires attention
                if request["cache_interval"] == 0:
                    # print("Shifting to attn queue")
                    await request_pool.attn_queue.put(request_id)
                else:
                    active_requests.append(request_id)

            # Re-populate active queue with remaining requests
            for request_id in active_requests:
                await request_pool.active_queue.put(request_id)

            # Step 2: Add new requests from the pool if attn_queue is not full
            for request_id, request in request_pool.requests.items():
                if request_pool.attn_queue.qsize() >= self.batch_size:
                    break  # Stop if attn_queue is full
                if request["status"] == RequestStatus.PENDING:
                    if (request_pool.active_queue.qsize() + request_pool.attn_queue.qsize()) < max_active_requests:
                        await request_pool.attn_queue.put(request_id)
                        request["status"] = RequestStatus.IN_PROGRESS


    async def shift_to_active_queue_from_attn(self, request_pool):
        """
        Transfer requests from the attention queue back to the active queue
        after attention-specific processing is complete.
        """
        async with self.lock:
            while not request_pool.attn_queue.empty():
                request_id = await request_pool.attn_queue.get()
                # Add the request back to the active queue
                await request_pool.add_to_active_queue(request_id)