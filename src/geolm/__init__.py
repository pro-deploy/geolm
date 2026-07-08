"""geolm: narrow geometric models. Train for one task, ship to inference."""
from .pipeline import GeoLM
from .model import GeoConfig, GeoModel
from .data import Vocab, read_jsonl

__version__ = "0.1.0"
__all__ = ["GeoLM", "GeoConfig", "GeoModel", "Vocab", "read_jsonl"]
