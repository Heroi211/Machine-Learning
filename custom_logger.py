import logging
import os
import sys
from configs import settings

def setup_log(snapshot_path, now):
    """
    Configura o logging para escrever no console e em um arquivo txt.
    """
  
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    log_filename = os.path.join(snapshot_path, f"pipeline_{now}.txt")
  
    log_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
 
    logger = logging.getLogger()
    logger.setLevel(settings.get_log_level())

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logging.info(f"💾 Log persistente configurado em: {log_filename}")
    return logger