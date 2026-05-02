import logging
from src.settings.config import BASE_DIR
from logging.handlers import RotatingFileHandler

def setup_logger():
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                log_dir / "app.log", 
                maxBytes=10**6, backupCount=3)
        ], force=True)

def get_logger(name): return logging.getLogger(name)