
from pathlib import Path
import os
from .settings import TrainingSettings
import yaml

def get_default_settings():
    """ Returns default settings."""
    return TrainingSettings()
        
def read_settings(filename: str):
    """ Reads settings from given yaml file"""
    try:
        with open(filename, "r") as f:
            args = yaml.safe_load(f)
            return TrainingSettings(**args)
    except FileNotFoundError as e:
        print(f"Settings file: {filename} is not found.")
        raise e
        
def save_settings(filename: str, settings: TrainingSettings):
    """Saves a settings to file."""
    with open(filename, "w") as f:
        yaml.safe_dump(settings.model_dump(), f)
