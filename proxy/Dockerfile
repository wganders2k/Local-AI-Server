FROM python:3.12-slim

WORKDIR /app

# nvidia-smi is *not* installed here. The NVIDIA container runtime injects it,
# along with the matching driver libraries, when the service requests the
# `utility` driver capability — see docker-compose.yml. Installing nvidia-utils
# in the image instead would pin a driver version that silently mismatches the
# host's after any upgrade.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 11436

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11436"]
