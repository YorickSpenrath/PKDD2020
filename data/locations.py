from pathlib import Path

CLOUD_DATA_FOLDER = None

try:
    from __local_settings import CLOUD_DATA_FOLDER
except ImportError:
    raise FileNotFoundError('Please create a variable called "CLOUD_DATA_FOLDER" in ./__local_settings.py to use'
                            ' cloud data')

CLOUD_DATA_FOLDER = Path(CLOUD_DATA_FOLDER)
