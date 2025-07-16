import asyncio
import os
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from serving.request_handler import RequestHandler
from serving.config_loader import ConfigLoader
from pipeline.pipeline import SD3Inferencer
from utils.logger import get_logger
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI



logger = get_logger(__name__)

# Load configuration
config_loader = ConfigLoader()
config = config_loader.config

# Initialize the RequestHandler
handler = None
inference_handler = None
os.environ["PROFILE_GPU"] = str(config["system"].get("profile_gpu", False)).lower()


# Initialize FastAPI app
app = FastAPI()


class RequestInput(BaseModel):
    prompt: str
    timesteps_left: int


@app.post("/start_background_process")
async def start_background_process(model, background_tasks: BackgroundTasks):
    global inference_handler
    background_tasks.add_task(handler.process_request, inference_handler, save_latents=False)
    return {"message": "Background process started."}


@app.post("/change_caching_interval")
async def change_caching_interval(cache_interval: int):
    global handler
    try:
        handler.cache_interval = cache_interval
        logger.info(f"Cache interval updated")
        return {"message": "Updated cache interval"}
    except Exception as e:
        logger.error(f"could not change interval")
        raise HTTPException(status_code=500, detail="Failed to change interval")


@app.post("/add_request")
async def add_request(request: RequestInput):
    """
    Add a new request to the system.
    """
    try:
        await handler.add_request(request.prompt, request.timesteps_left)
        logger.info(f"New request added: Timesteps={request.timesteps_left}")
        return {"message": "Request added successfully."}
    except Exception as e:
        logger.error(f"Error adding request: {e}")
        raise HTTPException(status_code=500, detail="Failed to add request.")


@app.post("/process_requests")
async def process_requests():
    """
    Start processing requests asynchronously.
    """
    try:
        model = None  # Replace with the actual model if needed
        asyncio.create_task(handler.process_request(model))
        logger.info("Request processing started.")
        return {"message": "Request processing started."}
    except Exception as e:
        logger.error(f"Error starting request processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start request processing.")


@app.get("/get_output")
async def get_output():
    """
    Retrieve completed requests from the output pool.
    """
    try:
        completed_requests = []
        while not handler.request_pool.output_pool.empty():
            request = await handler.request_pool.output_pool.get()
            time_completed = datetime.now().isoformat()
            request["time_completed"] = datetime.fromisoformat(time_completed) - datetime.fromisoformat(request["timestamp"])
            image_data = request.get("image", None)
            image_path = ""
            if image_data:
                output_dir = "output_images"
                os.makedirs(output_dir, exist_ok=True)
                image_path = os.path.abspath(os.path.join(output_dir, f"request_{request['request_id']}.png"))
                image_data.save(image_path)
            completed_requests.append(
                {
                    "request_id": request["request_id"], 
                    "prompt": request["prompt"], 
                    "elapsed_gpu_time": request["elapsed_gpu_time"],
                    "status": request["status"], 
                    "timestamp": request["timestamp"],
                    "processing_time_start": request.get("processing_time_start", 0),
                    "time_completed": time_completed, 
                    "image": image_path
                }
            )
            logger.info(f"Retrieved completed request: {request}")
        logger.info(f"Retrieved {len(completed_requests)} completed requests.")
        return {"completed_requests": completed_requests}
    except Exception as e:
        logger.error(f"Error retrieving completed requests: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve completed requests.")


# @app.on_event("startup")
# async def startup_event():
#     """
#     Start the background task to monitor and process requests.
#     """
#     try:
#         model = None  # Replace with the actual model if needed
#         asyncio.create_task(handler.process_request(model))
#         logger.info("Background request processing started during server startup.")
#     except Exception as e:
#         logger.error(f"Error during startup: {e}")
#         raise RuntimeError("Failed to initialize background processing.")


@app.on_event("startup")
async def startup_event():
    """
    Load the model and start the background task to monitor and process requests.
    """
    global inference_handler
    global handler
    try:
        logger.info("Loading model during startup...")
        inference_handler = SD3Inferencer()
        model_path = config["model"]["model_path"]
        model_folder = config["model"]["model_folder"]
        inference_handler.load(
            model=model_path,
            model_folder=model_folder,
            text_encoder_device="cuda",
            verbose=False,
            shift=5,
            custom_scheduler=True
        )
        handler = RequestHandler(config, inference_handler)
        logger.info("Background request processing started during server startup.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise RuntimeError("Failed to initialize background processing.")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
    