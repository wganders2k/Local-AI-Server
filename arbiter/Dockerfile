FROM python:3.12-slim

WORKDIR /app

# nvidia-smi is injected by the NVIDIA container runtime when the service asks
# for the `utility` driver capability — see docker-compose.yml. Installing
# nvidia-utils here instead would pin a driver version that silently mismatches
# the host's after any upgrade.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 11438

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11438"]
