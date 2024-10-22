import copy
from typing import Dict
from pipelines.base_pipeline_dit import DitPipeline
from diffusers import AutoencoderKL, DDIMScheduler
try:
    from models.dit import Transformer2DModelWithAttScores
except:
    import sys
    sys.path.insert(0, "../")
    from models.dit import Transformer2DModelWithAttScores

class DitPipelineAttnScores(DitPipeline):
    def __init__(
            self, 
            transformer: Transformer2DModelWithAttScores, 
            vae: AutoencoderKL, 
            scheduler: DDIMScheduler, 
            id2label: Dict[int, str] | None = None
        ):
        new_transformer = copy.deepcopy(transformer)
        new_transformer.__class__ = Transformer2DModelWithAttScores
        new_transformer.__init__(**transformer.config)
        new_transformer.load_state_dict(transformer.state_dict())
        super().__init__(new_transformer, vae, scheduler, id2label)
        