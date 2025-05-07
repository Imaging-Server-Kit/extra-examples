from typing import List
from pathlib import Path
import numpy as np
import uvicorn

from imaging_server_kit import algorithm_server, ImageUI, MaskUI, DropDownUI

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_napari_tracks

@algorithm_server(
    algorithm_name="trackastra",
    parameters={
        "image": ImageUI(),
        "mask": MaskUI(),
        "mode": DropDownUI(
            title="Mode",
            description="Tracking mode.",
            items=["greedy", "greedy_nodiv"],
            default="greedy",
        ),
    },
    sample_images=[
        Path(__file__).parent / "sample_images" / "trpL_150310-11_img.tif",
        Path(__file__).parent / "sample_images" / "trpL_150310-11_mask.tif",
    ],
    metadata_file="metadata.yaml",
)
def trackastra_server(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str,
) -> List[tuple]:
    """Runs the algorithm."""

    device = "cpu"  # For now - To avoid CUDA errors

    model = Trackastra.from_pretrained("general_2d", device=device)

    track_graph = model.track(
        image, mask, mode=mode
    )  # or mode="ilp", or "greedy_nodiv"

    napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)

    return [(napari_tracks, {"name": "Tracks"}, "tracks")]


if __name__ == "__main__":
    uvicorn.run(trackastra_server.app, host="0.0.0.0", port=8000)
