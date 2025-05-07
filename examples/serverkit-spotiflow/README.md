![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# serverkit-spotiflow

Implementation of a web server for [Spotiflow](https://github.com/weigertlab/spotiflow).

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

## Sample images provenance

- `hybiss_2d.tif`: Single test HybISS image from the Spotiflow paper (doi.org/10.1101/2024.02.01.578426).
