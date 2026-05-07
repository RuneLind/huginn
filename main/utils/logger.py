import logging
import sys
from pathlib import Path


def setup_root_logger(level: int = logging.INFO) -> logging.Logger:
    root_logger = logging.getLogger()
    
    # Avoid adding multiple handlers if already configured
    if root_logger.handlers:
        return root_logger
    
    root_logger.setLevel(level)
    
    # Create console handler for root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter for root logger
    formatter = logging.Formatter(
        '%(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Configure specific third-party library log levels
    configure_third_party_loggers()
    
    return root_logger

def configure_third_party_loggers():
    library_configs = {
        'faiss': logging.INFO,
        'sentence_transformers': logging.WARNING,
        'httpx': logging.WARNING,
        'huggingface_hub': logging.WARNING,
        'transformers': logging.ERROR,
        'transformers.utils.loading_report': logging.ERROR,
        'urllib3': logging.WARNING,
    }
    
    for lib_name, log_level in library_configs.items():
        logging.getLogger(lib_name).setLevel(log_level)


def route_handlers_to_stderr():
    """Switch every existing root-logger handler over to ``sys.stderr``.

    MCP stdio adapters use stdout for the MCP protocol, so any logging that
    landed there would corrupt the wire format.
    """
    try:
        for handler in logging.getLogger().handlers:
            handler.setStream(sys.stderr)
    except Exception:
        pass


def add_file_handler(filename: str, base_dir: Path | None = None):
    """Attach a file handler under ``~/logs/<filename>`` to the root logger.

    Silently no-ops if the file cannot be opened — adapters must keep running
    even if disk logging is unavailable.
    """
    try:
        log_dir = base_dir if base_dir is not None else (Path.home() / "logs")
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / filename)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)
    except Exception:
        pass 