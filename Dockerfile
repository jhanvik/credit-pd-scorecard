# NOT BUILT OR TESTED by the author's tooling: build it locally and fix any errors before relying on it.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which slim images do not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
