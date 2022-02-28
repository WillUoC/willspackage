# A module for custom errors used in package
## as per Python industry standards
import sys

class ArgumentError(Exception):
    """Raised for unexpected arguments are passed"""
    pass

class RootError(Exception):
    """The root could not be found for the given equation. Likely due to the root not being bracketed in given range."""
    pass