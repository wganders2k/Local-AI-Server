
***

# Implementation Plan: Migrate LLM Proxy to Prometheus/Grafana Monitoring

## Objective
Replace the custom, in-memory `job_history.py` metrics tracking system in our FastAPI LLM routing proxy with industry-standard Prometheus metrics. We will expose a `/metrics` endpoint, delete the old manual tracking code, and inject Prometheus and Grafana into the existing Docker Compose infrastructure.

## Phase 1: Dependency Management
1. **Update Requirements**: 
   - Add `prometheus-client` to `requirements.txt` (or `pyproject.toml` / Pipfile, depending on the project structure).
   - *Agent Instruction:* Ensure this dependency is placed where the proxy container's Dockerfile will install it during a `docker compose build`.

## Phase 2: Refactor `proxy.py` - Setup & Cleanup
**Target File:** `proxy.py`

1. **Remove Legacy Imports & Background Tasks**:
   - Delete the import: `from job_history import job_history`.
   - In the `lifespan` context manager, remove `heartbeat_task = asyncio.create_task(_job_history_heartbeat())` and its associated `cancel()` / `await` blocks.
   - Delete the `_job_history_heartbeat` async function entirely.
   - Delete the `@app.get("/history")` and `@app.get("/history/summary")` endpoints entirely.

2. **Import Prometheus Client**:
   - Add the following imports at the top of the file:
     ```python
     from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
     ```

3. **Define Prometheus Metrics**:
   - Define these global metrics variables near the top of the file (after imports and logger setup):
     ```python
     # Prometheus Metrics
     PROXY_REQUESTS = Counter("proxy_requests_total", "Total LLM requests", ["model", "status"])
     PROXY_ACTIVE_REQUESTS = Gauge("proxy_active_requests", "Currently processing requests", ["model"])
     PROXY_TOKENS = Counter("proxy_tokens_total", "Total tokens processed", ["model", "type"]) # type="input" or "output"
     PROXY_QUEUE_DEPTH = Gauge("proxy_queue_depth", "Current requests waiting for a model swap")
     PROXY_EXTERNAL_JOBS = Gauge("proxy_external_jobs_active", "Number of active external VRAM jobs")
     GPU_VRAM_USED = Gauge("gpu_vram_used_bytes", "Used GPU VRAM in bytes")
     GPU_VRAM_TOTAL = Gauge("gpu_vram_total_bytes", "Total GPU VRAM in bytes")
     PROXY_REQUEST_DURATION = Histogram("proxy_request_duration_seconds", "Total request time", ["model"])
     ```

## Phase 3: Refactor `proxy.py` - Core Logic Updates
**Target File:** `proxy.py`

1. **Update `_vram_monitor`**:
   - Modify the parsing logic in `_vram_monitor` to update the Prometheus gauges. Convert MiB to bytes.
   - *Agent Instruction:* Replace the `logger.info` VRAM block with:
     ```python
     used_mib, total_mib = (float(x.strip()) for x in line.split(","))
     GPU_VRAM_USED.set(used_mib * 1024 * 1024)
     GPU_VRAM_TOTAL.set(total_mib * 1024 * 1024)
     # Optional: Keep the logger.info if console output is still desired
     ```

2. **Update the `_forward` method**:
   - **Start of request:** Remove `job_history.request_start(model)`. Replace it with:
     ```python
     label = model or "unknown_model"
     PROXY_ACTIVE_REQUESTS.labels(model=label).inc()
     ```
   - **Error Handling (Before Stream):** Wrap the `await _http_client.send()` in a `try/except` block to catch immediate connection failures, decrementing active requests and logging the error status to `PROXY_REQUESTS`.
   - **End of request (`_stream_with_timing` finally block):** 
     - Remove the `job_history.request_end(...)` call. 
     - Insert the following Prometheus metric updates:
       ```python
       if output_tokens == 0 and full_content:
           output_tokens = max(1, len("".join(full_content)) // 4)
           
       PROXY_TOKENS.labels(model=label, type="input").inc(input_tokens)
       PROXY_TOKENS.labels(model=label, type="output").inc(output_tokens)
       PROXY_REQUESTS.labels(model=label, status=str(response.status_code)).inc()
       PROXY_REQUEST_DURATION.labels(model=label).observe(time.monotonic() - t_start)
       
       PROXY_ACTIVE_REQUESTS.labels(model=label).dec()
       ```

3. **Add the `/metrics` Endpoint**:
   - Add this new route to expose metrics for scraping:
     ```python
     @app.get("/metrics")
     async def metrics():
         PROXY_QUEUE_DEPTH.set(state.queue_depth)
         PROXY_EXTERNAL_JOBS.set(len(external_jobs.active_job_ids))
         return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
     ```

## Phase 4: Delete Legacy Code
**Target File:** `job_history.py`

1. **Delete File**: Safely delete `job_history.py` entirely. It is no longer needed.

## Phase 5: Infrastructure (Docker Compose Integration)
**Target Files:** `docker-compose.yml` (or `compose.yaml`), `prometheus.yml`

1. **Analyze Existing Docker Compose**:
   - *Agent Instruction:* Read the user's existing `docker-compose.yml`. Identify the exact service name and internal port used for the FastAPI proxy (e.g., `proxy:8000`, `api:8080`, etc.).

2. **Create `prometheus.yml`**:
   - Create a file named `prometheus.yml` in the project root.
   - *Agent Instruction:* Use the exact service name and port discovered in Step 1 for the target.
     ```yaml
     global:
       scrape_interval: 5s

     scrape_configs:
       - job_name: 'llm-proxy'
         static_configs:
           - targets: ['<DISCOVERED_PROXY_SERVICE_NAME>:<DISCOVERED_PORT>'] 
     ```

3. **Inject Monitoring Services into Docker Compose**:
   - *Agent Instruction:* Add the `prometheus` and `grafana` services to the existing compose file. Do NOT overwrite or remove existing services (like the llama-servers or the proxy). Add a `grafana_data` volume so dashboards are saved persistently.
     ```yaml
     # Add this under the services block:
       prometheus:
         image: prom/prometheus:latest
         volumes:
           - ./prometheus.yml:/etc/prometheus/prometheus.yml
         ports:
           - "9090:9090"
         restart: unless-stopped

       grafana:
         image: grafana/grafana:latest
         ports:
           - "3000:3000"
         environment:
           - GF_SECURITY_ADMIN_PASSWORD=admin # Default password
         volumes:
           - grafana_data:/var/lib/grafana
         restart: unless-stopped

     # Add this at the bottom of the file:
     volumes:
       grafana_data:
     # (If a volumes block already exists, just append grafana_data: to it)
     ```

## Phase 6: Post-Implementation Verification
*Agent Instruction: Print this verification and setup list to the user once coding is complete.*

1. Apply the changes and rebuild the proxy container to install the new dependency:
   ```bash
   docker compose down
   docker compose up -d --build
   ```
2. Verify the metrics endpoint is responding (from your host machine):
   ```bash
   curl http://localhost:<PROXY_EXTERNAL_PORT>/metrics
   ```
   *(You should see `proxy_requests_total`, `gpu_vram_used_bytes`, etc.)*
3. Open Grafana in your browser: `http://localhost:3000` (login: `admin` / `admin`).
4. Navigate to **Connections -> Add Data Source -> Prometheus**.
   - Set the Prometheus server URL to: `http://prometheus:9090` (Docker will resolve this internally).
   - Click **Save & Test**.
5. **Create a Dashboard with these PromQL Queries:**
   - **Active Requests:** `proxy_active_requests`
   - **Rolling Tokens/Sec (TPS):** `rate(proxy_tokens_total{type="output"}[1m])`
   - **VRAM Usage %:** `gpu_vram_used_bytes / gpu_vram_total_bytes`
   - **Total Requests (last 24h):** `increase(proxy_requests_total[24h])`
   - **Queue Depth:** `proxy_queue_depth`