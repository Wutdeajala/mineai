from dotenv import load_dotenv
import os

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

if __name__ == "__main__":
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Base URL: {OLLAMA_BASE_URL}")