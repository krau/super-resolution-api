FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
COPY . /sr_api
WORKDIR /sr_api
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && uv sync --frozen --no-dev
ENTRYPOINT [ "uv" ,"run", "python", "main.py" ]