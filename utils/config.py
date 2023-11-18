import yaml
import json
from pathlib import Path

class Config:
    _instance = None
    _data = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path):
        self.path = Path(path)

    def get_config_dict(self):
        if not Config._data:
            if self.path.suffix == '.json':
                with open(self.path) as file:
                    Config._data = json.load(file)
            elif self.path.suffix == '.yaml':
                try:
                    with open(self.path, 'r') as f:
                        Config._data = yaml.safe_load(f)
                except yaml.parser.ParserError as e:
                    print("YAML file parsing error:", e)
            else:
                raise NotImplementedError
        return Config._data
