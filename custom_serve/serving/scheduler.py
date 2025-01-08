import asyncio

class Scheduler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.lock = asyncio.Lock()

    async def add_to_active_request_queue(self, request_pool, request_id):
        async with self.lock: 
            await request_pool.add_to_active_queue(request_id)

    async def shift_to_attn_queue(self, request_pool):
        async with self.lock:
            active_requests = []
            
            # Step 1: Process active queue first
            while not request_pool.active_queue.empty():
                request_id = await request_pool.active_queue.get()
                request = request_pool.requests[request_id]

                # Check if the request requires attention
                if request["cache_interval"] == 0:
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
                if request["status"] == "pending":
                    await request_pool.attn_queue.put(request_id)
                    request["status"] = "in_progress"

            