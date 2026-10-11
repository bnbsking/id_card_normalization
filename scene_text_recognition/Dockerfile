FROM python:3.12-slim

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/_externals/sdk/OpenOCR
ENV PIP_DEFAULT_TIMEOUT=86400

RUN apt update && apt install -y \
    vim \
    curl \
    procps \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    build-essential

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY . /app

RUN poetry install --no-root --no-interaction --no-ansi
RUN poetry run pip install -r /app/_externals/sdk/OpenOCR/requirements.txt
RUN poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
RUN poetry run pip install 'numpy<2'
RUN poetry run pip install 'scipy<1.12'
