from typing import List
from pathlib import Path
import numpy as np
import uvicorn
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity

from imaging_server_kit import algorithm_server, ImageUI, PointsUI, ShapesUI, BoolUI

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = SAM2ImagePredictor.from_pretrained(
    "facebook/sam2-hiera-tiny", device=DEVICE
)
generator = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-tiny", device=DEVICE
)


@algorithm_server(
    algorithm_name="sam2",
    parameters={
        "image": ImageUI(description="Input image (2D, RGB)."),
        "boxes": ShapesUI(
            title="Boxes",
            description="Boxes prompt.",
        ),
        "points": PointsUI(
            title="Points",
            description="Points prompt.",
        ),
        "auto_mode": BoolUI(
            default=False,
            title="Auto mode",
            description="Run SAM in auto (grid) mode",
        ),
    },
    sample_images=[
        Path(__file__).parent / "sample_images" / "groceries.jpg",
    ],
    metadata_file="metadata.yaml",
)
def sam2_server(
    image: np.ndarray,
    boxes: np.ndarray,
    points: np.ndarray,
    auto_mode: bool,
) -> List[tuple]:
    """Runs the algorithm."""
    if len(image.shape) == 2:
        image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        image = gray2rgb(image)

    rx, ry, _ = image.shape
    segmentation = np.zeros((rx, ry), dtype=np.uint16)

    if auto_mode:
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            masks = generator.generate(image)

        for k, ann in enumerate(masks):
            m = ann.get("segmentation")
            segmentation[m] = k + 1
    else:
        # Boxes should be in (N, 4) format (top-left, bottom-right corner)
        # Invert X-Y and keep only two vertices
        # Note - right now, this can cause bugs if the boxes arent drawn in the top to bottom direction
        if len(boxes):
            print(f"{boxes=}")
            boxes = boxes[:, ::2, ::-1].reshape((len(boxes), -1)).copy()
        else:
            boxes = None

        if len(points):
            point_labels = np.ones(len(points))
            points = points[:, ::-1].copy()  # Invert X-Y
        else:
            points = None
            point_labels = None

        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=boxes,
                multimask_output=False,
            )

        for k, m in enumerate(masks):
            segmentation[np.squeeze(m) == 1] = k + 1

    return [(segmentation, {"name": "SAM-2 result"}, "instance_mask")]


if __name__ == "__main__":
    uvicorn.run(sam2_server, host="0.0.0.0", port=8000)
