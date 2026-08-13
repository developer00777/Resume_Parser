FROM python:3.11-slim

# LLM model used by this service
LABEL llm.model="openai/gpt-4o-mini" \
      llm.provider="openrouter" \
      maintainer="Resume Parser API"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Copy tests and generate sample resume PDF
COPY tests/ tests/
RUN python tests/generate_sample_pdf.py

EXPOSE 8000

# --workers 1 is deliberate, not a default. The async /parse/job endpoints keep
# their job store in a process-local dict, so a second worker would serve
# "job not found" for jobs the first one is running. Scale with more instances
# behind a load balancer only after moving that store to Redis
# (docs/REMAINING_WORK.md item 3). The Salesforce write-back path is stateless
# and unaffected either way.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
