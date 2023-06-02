class WolfLasError(Exception):
    """Base WolfLas exception that all others inherit"""


class InvalidClassError(WolfLasError):
    """Called when a specified classification number isn't found"""


class ScanError(WolfLasError):
    """Called when o3d's dbscan method throws an error"""


class VersionError(WolfLasError):
    """Called when an incorrect las version is given"""
