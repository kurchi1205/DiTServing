import Path
import uuid
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from infer_base_pipeline import load_pipe, generate

IMAGE_SAVE_PATH = Path("./generated_images")
IMAGE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
pipe = None

class RequestModel(BaseModel):
    prompt: str
    num_inference_steps: int

app = FastAPI()

@app.on_event("startup")
def load_inference_pipeline():
    global pipe
    pipe_config_arg = {
        "pipeline_type": "base"
    }
    pipe = load_pipe(pipe_config_arg)  # Load the DiT model pipeline during startup
    print("Pipeline loaded successfully.")

@app.post("/generate/")
def inference(request: RequestModel):
    try:
        args = request.dict()
        image = generate(pipe, **args)
        filename = f"{uuid.uuid4()}.jpeg"
        image_path = str(IMAGE_SAVE_PATH / filename)
        image.save(image_path)
        return {"message": f"Image saved at {image_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))