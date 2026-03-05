from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
