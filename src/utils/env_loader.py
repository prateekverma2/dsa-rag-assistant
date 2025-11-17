# src/utils/env_loader.py
import os
from dotenv import load_dotenv

def load_environment():
    """
    Loads environment variables from the .env file into os.environ.
    Call this once at program startup (before any API calls).
    """
    # Find .env file two levels up (from src/utils/)
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")

    if not os.path.exists(env_path):
        raise FileNotFoundError(f"❌ .env file not found at {env_path}")

    load_dotenv(dotenv_path=env_path, override=True)

    print("✅ Environment loaded successfully")
    print(f"   Pinecone Index  → {os.getenv('PINECONE_INDEX_NAME')}")
    print(f"   LLM Provider    → {os.getenv('LLM_PROVIDER')}")
