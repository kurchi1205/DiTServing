import yaml
from pathlib import Path


class ConfigLoader:
    def __init__(self, config_path="../configs/config.yaml"):
        config_path = (Path(__file__).parent.parent / "configs/config.yaml")
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        """Get a value from the config with a default fallback."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                break
        return value
