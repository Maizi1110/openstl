"""Compatibility shim for exPreCast implementations.

The repository's canonical implementation currently lives in exprecast_model.py.
This shim keeps `openstl.models.exprecast` import paths working.
"""

from .exprecast_model import *  # noqa: F401,F403
