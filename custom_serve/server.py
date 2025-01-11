import asyncio
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from serving.request_handler import RequestHandler
from serving.config_loader import ConfigLoader
from utils.logger import get_logger
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI



logger = get_logger(__name__)

# Load configuration
config_loader = ConfigLoader()
config = config_loader.config

# Initialize the RequestHandler
handler = RequestHandler(config)

# Initialize FastAPI app
app = FastAPI()


class RequestInput(BaseModel):
    prompt: str
    timesteps_left: int


@app.post("/start_background_process")
async def start_background_process(model, background_tasks: BackgroundTasks):
    background_tasks.add_task(handler.process_request, model)
    return {"message": "Background process started."}


@app.post("/add_request")
async def add_request(request: RequestInput):
    """
    Add a new request to the system.
    """
    try:
        await handler.add_request(request.prompt, request.timesteps_left)
        logger.info(f"New request added: Prompt={request.prompt}, Timesteps={request.timesteps_left}")
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
            completed_requests.append(
                {"request_id": request["request_id"], "prompt": request["prompt"], "status": request["status"], "timestamp": request["timestamp"], "time_completed": time_completed}
            )
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



# @app.on_event("shutdown")
# async def shutdown_event():
#     """
#     Clean up resources and log shutdown details.
#     """
#     try:
#         logger.info("Server is shutting down. Cleaning up resources.")
#         # Add any necessary cleanup code here (e.g., releasing model resources)
#     except Exception as e:
#         logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")