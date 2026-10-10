"""Decoder-benchmark datasets: on-disk format and I/O (see schema.py)."""
from .schema import SCHEMA_VERSION, read_dataset, write_dataset

__all__ = ["SCHEMA_VERSION", "read_dataset", "write_dataset"]
