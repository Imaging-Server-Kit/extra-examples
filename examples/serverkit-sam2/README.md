![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# serverkit-sam2

Implementation of a web server for [SAM 2](https://github.com/facebookresearch/sam2).

## Installing the algorithm server with `pip`

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
python main.py
```

The server will be running on http://localhost:8000.

## Using `docker-compose`

To build the docker image and run a container for the algorithm server in a single command, use:

```
docker compose up
```

The server will be running on http://localhost:8000.

## Sample images provenance -->

- `groceries.jpg`: from the original sam2 repository.
