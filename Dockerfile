FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_DATA_DIR=/home/assistant/.agent_data

WORKDIR /app

COPY requirements.txt pyproject.toml contract.py ./
RUN pip install --no-cache-dir -r requirements.txt
COPY cli ./cli
COPY agent ./agent
COPY reviewer ./reviewer
COPY mcp_server ./mcp_server
COPY observability ./observability
COPY cli_design.toml ./

RUN python -m reviewer.init_sources \
    && pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 assistant \
    && mkdir -p /home/assistant/.agent_data \
    && chown -R assistant:assistant /app /home/assistant/.agent_data

USER assistant

ENTRYPOINT ["bank-agent"]
CMD ["chat"]
