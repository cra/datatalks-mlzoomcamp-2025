FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /bin/uv
COPY pyproject.toml uv.lock .

RUN uv sync --frozen --no-install-project

ENV MODELV1_PATH=/code/pipeline_v1.bin
ENV MODELV2_PATH=/code/pipeline_v2.bin

COPY ./pipeline_v1.bin .
COPY ./src/server/main.py .
COPY ./src/server/__init__.py .

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
