import os
from src.utils.config_loader import load_project_config
from src.indexer.indexer import Indexer

if __name__ == "__main__":
    cfg = load_project_config("configs/config.yaml")
    Indexer("configs/config.yaml").build_index()
