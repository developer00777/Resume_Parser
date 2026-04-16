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

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
