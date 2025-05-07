# Doc: https://github.com/conda-forge/miniforge-images
FROM condaforge/miniforge3:latest

ARG PYTHON_VERSION=3.10
RUN conda install python=$PYTHON_VERSION -y

WORKDIR /app

COPY . .

ENV SAM2_REPO_ROOT=./sam2
ENV PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"

RUN python -m pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]

