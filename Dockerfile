FROM python:3.12.13-slim

# uv, to install the exact library versions from uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /code

# install the dependencies first, so Docker can reuse this layer when only the code changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
