FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Beside the modules that import it by bare name (see docker-compose.yml).
COPY --from=shared mimic_format.py persona_roster.py mimic_persona.md ./

CMD ["python", "-u", "bot.py"]
