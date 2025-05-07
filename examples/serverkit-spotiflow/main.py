from typing import List
from pathlib import Path
import numpy as np
import uvicorn
from spotiflow.model import Spotiflow

from imaging_server_kit import algorithm_server, ImageUI


@algorithm_server(
    algorithm_name="spotiflow",
    parameters={"image": ImageUI()},
    sample_images=[
        Path(__file__).parent / "sample_images" / "hybiss_2d.tif",
    ],
    metadata_file="metadata.yaml",
)
def spotiflow_server(image: np.ndarray) -> List[tuple]:
    """Runs the algorithm."""
    model = Spotiflow.from_pretrained("general", map_location="cpu")

    points, details = model.predict(image)

    return [(points, {"name": "Detected spots"}, "points")]


if __name__ == "__main__":
    uvicorn.run(spotiflow_server.app, host="0.0.0.0", port=8000)
