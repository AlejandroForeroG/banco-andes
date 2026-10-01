FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_DATA_DIR=/home/agente/.agent_data

WORKDIR /app

COPY requirements.txt pyproject.toml contract.py ./
RUN pip install --no-cache-dir -r requirements.txt
COPY cli ./cli
COPY agent ./agent
COPY agente2 ./agente2
COPY mcp_server ./mcp_server
COPY observability ./observability
COPY cli_design.toml ./

RUN python -m agente2.montar_fuentes \
    && pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 agente \
    && mkdir -p /home/agente/.agent_data \
    && chown -R agente:agente /app /home/agente/.agent_data

USER agente

ENTRYPOINT ["agente"]
CMD ["chat"]
