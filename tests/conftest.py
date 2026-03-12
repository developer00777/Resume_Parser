import os

# Skip Ollama startup check during tests
os.environ["TESTING"] = "1"
