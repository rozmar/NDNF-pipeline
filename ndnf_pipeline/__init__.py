"""
NDNF Pipeline

A shared DataJoint pipeline for the Neuronal Diversity in Network Function lab.
Contains common schemas for metadata of subjects, surgeries, viruses etc.
Contains metadata updating from google spreadsheets.
"""

__version__ = "0.1.0"

# Import main modules for easier access
from . import lab
from . import utils

# Export key components
__all__ = [
    "lab",
    "utils",
]
