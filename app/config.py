from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenRouter ────────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    # Text extraction (3 parallel chunks) — DeepSeek V4 Flash: fast MoE, 70% cheaper output
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    # OCR fallback for image-based PDFs — must stay multimodal (vision-capable)
    openrouter_ocr_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Application ───────────────────────────────────────────────────────────
    api_key: str = "changeme"
    max_file_size: int = 10485760  # 10 MB
    log_level: str = "INFO"

    # ── Salesforce Connected App ──────────────────────────────────────────────
    # Required for /api/v1/salesforce/* endpoints.
    # Create a Connected App in Salesforce Setup → App Manager.
    sf_client_id: str = ""          # Consumer Key
    sf_client_secret: str = ""      # Consumer Secret

    # Optional: only needed for Username-Password OAuth flow
    sf_username: str = ""
    sf_password: str = ""
    sf_security_token: str = ""     # Append to password for IP-unrestricted orgs

    sf_login_url: str = "https://login.salesforce.com"  # use test.salesforce.com for sandbox
    sf_api_version: str = "59.0"

    # ── Synchronous bulk endpoints ────────────────────────────────────────────
    # Kept for small batches. Salesforce caps a synchronous callout at 120s, so
    # the budget below must stay under that. When the budget is hit the request
    # now returns 200 with whatever finished (partial results) instead of 504.
    bulk_max_files: int = 15
    bulk_timeout: float = 110.0
    bulk_concurrency: int = 5

    # ── Redis job queue (async bulk path) ─────────────────────────────────────
    # Railway's Redis plugin injects REDIS_URL automatically.
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "resume_parse"
    # Files accepted per async batch — no wall-clock limit on this path.
    queue_max_files: int = 100
    # Resumes parsed in parallel per worker process. Scale out by adding workers.
    worker_concurrency: int = 8
    # Per-file budget inside the worker (seconds). The worker records a failed
    # item when this is hit, so a batch can never hang on one bad file.
    job_timeout: int = 300
    # Attempts per file before it is recorded as failed (transient errors only).
    job_max_tries: int = 3
    # How long uploaded bytes and batch results live in Redis (seconds).
    job_ttl: int = 86400
    # A batch with no progress and nothing running for this long is reaped and
    # its unfinished files are marked failed, so callers always reach a terminal state.
    job_stale_seconds: int = 1800

    # ── Optional completion callback ──────────────────────────────────────────
    # When a submit request carries callback_url, the finished batch is POSTed there.
    job_callback_timeout: float = 30.0
    # Comma-separated host suffixes allowed as callback targets. Empty = allow any.
    job_callback_allowed_hosts: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
