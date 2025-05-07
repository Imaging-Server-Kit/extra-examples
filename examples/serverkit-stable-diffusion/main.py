from typing import List
import numpy as np
import uvicorn

from imaging_server_kit import algorithm_server, StringUI

import torch
from diffusers import StableDiffusionPipeline


@algorithm_server(
    algorithm_name="stable-diffusion",
    parameters={
        "prompt": StringUI(
            title="Prompt",
            description="Text prompt",
            default="An astronaut riding a horse on the moon.",
        )
    },
    metadata_file="metadata.yaml",
)
def stable_diffusion_server(prompt: str) -> List[tuple]:
    """Runs the algorithm."""

    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    generated_image = pipe(prompt).images[0]

    generated_image = np.asarray(generated_image)

    return [(generated_image, {}, "image")]


if __name__ == "__main__":
    uvicorn.run(stable_diffusion_server.app, host="0.0.0.0", port=8000)
