# src/utils/config_loader.py
import os
import yaml
from dotenv import load_dotenv

def load_project_config(config_path="configs/config.yaml"):
    """
    ✅ Universal loader:
    - Loads .env file (API keys, env vars)
    - Reads YAML config with UTF-8 encoding
    - Replaces ${VAR} placeholders with .env values
    Returns merged config dict
    """
    # Step 1: Load .env
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        print("⚠️ Warning: .env file not found. Using system environment variables.")

    # Step 2: Read config.yaml safely (UTF-8)
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = f.read()

    # Step 3: Replace ${VAR} with env values
    for key, val in os.environ.items():
        raw_config = raw_config.replace(f"${{{key}}}", val)

    # Step 4: Parse YAML
    try:
        config = yaml.safe_load(raw_config)
    except yaml.YAMLError as e:
        raise Exception(f"❌ Error parsing YAML config: {e}")

    print("✅ Config + Environment loaded successfully")
    return config
