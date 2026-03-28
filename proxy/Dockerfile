FROM python:3.12-slim

WORKDIR /app

# Install nvidia-utils for nvidia-smi (VRAM monitoring).
# Bind-mount /usr/bin/nvidia-smi from the host instead of installing
# the full driver package — the host already has nvidia-utils-580.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 11436

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11436"]
