import os

# Skip Ollama startup check during tests
os.environ["TESTING"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--count", action="store", default="2",
        help="Number of resumes per category for accuracy tests (0 = all)"
    )
    parser.addoption(
        "--concurrency", action="store", default="3",
        help="Max parallel LLM calls during accuracy test"
    )
