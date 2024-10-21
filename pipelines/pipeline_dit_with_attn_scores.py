from typing import Dict
from base_pipeline_dit import DitPipeline
from diffusers import AutoencoderKL, DDIMScheduler
try:
    from models.dit import DiTTransformer2DModelWithAttScores
except:
    import sys
    sys.path.insert(0, "../")
    from models.dit import DiTTransformer2DModelWithAttScores

class DitPipelineAttnScores(DitPipeline):
    def __init__(
            self, 
            transformer: DiTTransformer2DModelWithAttScores, 
            vae: AutoencoderKL, 
            scheduler: DDIMScheduler, 
            id2label: Dict[int, str] | None = None
        ):
        super().__init__(transformer, vae, scheduler, id2label)
        