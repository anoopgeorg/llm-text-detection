import os
import sys
import logging
from pathlib import Path

logging_str = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"
log_dir = Path("logs")
log_file_path = log_dir / "running_logs.log"

os.makedirs(str(log_dir), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
)


logger = logging.getLogger("llm-text-detection-logger")
