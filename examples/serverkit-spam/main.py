from typing import List
from pathlib import Path
import numpy as np
import uvicorn
import spam.DIC

from imaging_server_kit import algorithm_server, ImageUI


@algorithm_server(
    algorithm_name="spam-register",
    parameters={
        "moving_image": ImageUI(
            title="Moving image",
            description="Moving image (2D, 3D).",
            dimensionality=[2, 3],
        ),
        "fixed_image": ImageUI(
            title="Fixed image",
            description="Fixed image (2D, 3D).",
            dimensionality=[2, 3],
        ),
    },
    sample_images=[
        Path(__file__).parent / "sample_images" / "image1.tif",
        Path(__file__).parent / "sample_images" / "image2.tif",
    ],
    metadata_file="metadata.yaml",
)
def spam_register_server(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
) -> List[tuple]:
    """Runs the spam-reg algorithm."""
    reg = spam.DIC.register(moving_image, fixed_image)
    phi = reg.get("Phi")
    registered_image = spam.DIC.applyPhiPython(moving_image, Phi=phi)

    image_params = {"name": "Registered image"}

    return [(registered_image, image_params, "image")]


if __name__ == "__main__":
    uvicorn.run(spam_register_server.app, host="0.0.0.0", port=8000)
