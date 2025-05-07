from typing import List
from pathlib import Path
import numpy as np
import uvicorn

from imaging_server_kit import algorithm_server, ImageUI, StringUI
from transformers import BlipProcessor, BlipForConditionalGeneration


@algorithm_server(
    algorithm_name="blip-captioning",
    parameters={
        "image": ImageUI(description="Input image (2D, RGB)."),
        "conditional_text": StringUI(
            default="an image of",
            title="Text",
            description="Conditional text (beginning of the caption).",
        ),
    },
    sample_images=[Path(__file__).parent / "sample_images" / "astronaut.tif"],
    metadata_file="metadata.yaml",
)
def blip_server(
    image: np.ndarray,
    conditional_text: str,
) -> List[tuple]:
    """Runs the algorithm."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    inputs = processor(image, conditional_text, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return [(caption, {}, "text")]


if __name__ == "__main__":
    uvicorn.run(blip_server.app, host="0.0.0.0", port=8000)
