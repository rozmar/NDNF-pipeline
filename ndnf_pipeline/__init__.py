"""
NDNF Pipeline

A shared DataJoint pipeline for the Neuronal Diversity in Network Function lab.
Contains common schemas for metadata of subjects, surgeries, viruses etc.
Contains metadata updating from google spreadsheets.
"""

__version__ = "0.1.0"

# `lab` is intentionally not imported here: importing it opens a live DataJoint
# connection (schema activation), which used to fire as a side effect of
# importing *any* ndnf_pipeline submodule - including pure, DB-free code like
# ndnf_pipeline.plot.videography_plots. That broke ProcessPoolExecutor workers
# on Windows: each spawned worker re-imports the package fresh with no DB
# credentials loaded and no interactive stdin, so dj.conn()'s username prompt
# raised EOFError and crashed the whole pool. Import `lab` explicitly
# (`from ndnf_pipeline import lab`) wherever it's actually needed.
from . import utils

# Export key components
__all__ = [
    "utils",
]
