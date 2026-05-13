# python:3.11-slim pinned 2026-05-13 — update periodically
FROM python:3.11-slim@sha256:a5b427ace4900267d93db34138e512325c6fa6af84ad5e4ed5f3b36258cc4142

WORKDIR /app

# Install system dependencies.
# Download Docker CLI and Compose plugin as static binaries (no daemon, no containerd bloat).
ARG TARGETARCH
ARG DOCKER_VERSION=28.5.2
ARG COMPOSE_VERSION=2.39.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        DOCKER_ARCH="aarch64"; \
        COMPOSE_ARCH="aarch64"; \
    elif [ "$TARGETARCH" = "amd64" ] || [ -z "$TARGETARCH" ]; then \
        DOCKER_ARCH="x86_64"; \
        COMPOSE_ARCH="x86_64"; \
    else \
        echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi && \
    curl -fsSL "https://download.docker.com/linux/static/stable/${DOCKER_ARCH}/docker-${DOCKER_VERSION}.tgz" | \
        tar xz --strip-components=1 -C /usr/local/bin docker/docker && \
    mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -fsSL "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-${COMPOSE_ARCH}" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Optimize layer caching by installing dependencies before copying source code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

COPY --chown=appuser:appuser . .

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11437/health')" || exit 1

CMD ["python", "-u", "main.py"]