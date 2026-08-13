from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenRouter ────────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
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

    # Comma-separated hosts, besides the org itself, that /salesforce/parse-url
    # may fetch a resume from. The Salesforce token is NEVER sent to these —
    # the list only decides which hosts are reachable. Empty means
    # Salesforce-hosted resumes only.
    sf_external_resume_hosts: str = ""

    # How long the org's field mapping stays trusted before it is re-fetched.
    # Bounds how long an admin's new field or picklist value goes unseen.
    # 0 disables refreshing: the mapping is then only rebuilt at startup or on
    # an explicit /describe/reload.
    sf_describe_ttl_minutes: int = 60

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
