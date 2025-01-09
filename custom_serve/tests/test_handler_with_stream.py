import asyncio
import sys
from random import randint

sys.path.append("serving/")
from request_handler import RequestHandler


async def generate_streaming_requests(handler):
    """
    Continuously send a stream of new requests to the handler.
    """
    prompt_templates = [
        "Generate image of a cat",
        "Generate image of a dog",
        "Generate image of a car",
        "Generate image of a landscape",
        "Generate image of a bird",
    ]

    while True:
        # Randomly select a prompt and timesteps
        prompt = prompt_templates[randint(0, len(prompt_templates) - 1)]
        timesteps_left = randint(10, 50)  # Random timesteps between 10 and 50
        
        # Add the request to the handler
        await handler.add_request(prompt, timesteps_left)
        print(f"New Request Added: Prompt={prompt}, Timesteps={timesteps_left}")
        
        # Wait before sending the next request
        await asyncio.sleep(1)  # New request every second


async def test_request_handler():
    # Initialize the RequestHandler and DummyModel
    handler = RequestHandler()
    model = None

    # List of initial prompts
    initial_prompts = [
        {"prompt": "Generate image of a cat", "timesteps_left": 30},
        {"prompt": "Generate image of a dog", "timesteps_left": 30},
        {"prompt": "Generate image of a car", "timesteps_left": 30},
    ]

    # Add initial requests to the handler
    for request in initial_prompts:
        await handler.add_request(request["prompt"], request["timesteps_left"])

    # Run the handler's processing loop asynchronously
    process_task = asyncio.create_task(handler.process_request(model))
    streaming_task = asyncio.create_task(generate_streaming_requests(handler))

    # Run for a fixed duration to allow processing
    await asyncio.wait([process_task, streaming_task], timeout=60)


# Run the test
if __name__ == "__main__":
    asyncio.run(test_request_handler())
