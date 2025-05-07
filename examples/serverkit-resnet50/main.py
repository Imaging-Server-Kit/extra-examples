from typing import List
from pathlib import Path
import numpy as np
import uvicorn
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification

from imaging_server_kit import algorithm_server, ImageUI


@algorithm_server(
    algorithm_name="resnet50",
    parameters={"image": ImageUI(description="Input image (2D, RGB)")},
    sample_images=[Path(__file__).parent / "sample_images" / "astronaut.tif"],
    metadata_file="metadata.yaml",
)
def resnet_server(image: np.ndarray) -> List[tuple]:
    """Runs the algorithm."""
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_label_idx]

    return [(predicted_label, {}, "class")]


if __name__ == "__main__":
    uvicorn.run(resnet_server.app, host="0.0.0.0", port=8000)
