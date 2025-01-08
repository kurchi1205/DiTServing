import asyncio
import sys
sys.path.append("serving/")
from request_handler import RequestHandler

async def log_completed_requests(request_pool):
    """
    Continuously check the request pool and log completed requests.
    """
    while True:
        completed_requests = []
        async with request_pool.lock:
            for request_id, request in list(request_pool.requests.items()):
                if request["status"] == "completed":
                    completed_requests.append(request_id)
        
        for request_id in completed_requests:
            # Log the completed request
            request = request_pool.requests.pop(request_id)
            print(f"Request Completed: ID={request_id}, Prompt={request['prompt']}")
        
        await asyncio.sleep(0.1)  # Avoid high CPU usage


async def test_request_handler():
    # Initialize the RequestHandler and DummyModel
    handler = RequestHandler()
    model = None

    # List of prompts and their respective timesteps
    prompts = [
        {"prompt": "Generate image of a cat", "timesteps_left": 30},
        {"prompt": "Generate image of a dog", "timesteps_left": 30},
        {"prompt": "Generate image of a car", "timesteps_left": 30},
    ]

    # Add requests to the handler
    for request in prompts:
        handler.add_request(request["prompt"], request["timesteps_left"])

    # Run the handler's processing loop asynchronously
    process_task = asyncio.create_task(handler.process_request(model))
    log_task = asyncio.create_task(log_completed_requests(handler.request_pool))

    # Run for a fixed duration to allow processing
    await asyncio.wait([process_task, log_task], timeout=10)

# Run the test
if __name__ == "__main__":
    asyncio.run(test_request_handler())
