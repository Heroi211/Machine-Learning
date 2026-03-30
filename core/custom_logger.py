import logging
import os
import sys
from core.configs import settings
from core.custom_logger import setup_log
from datetime import datetime
from typing import Dict, Any
import logging

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

def log_requests(
    now:str, 
    request_id: str,
    request_payload: str,
    response_payload: Dict[str, Any],
    status_code: int,
    latency_ms:float,
     ):
    
    try:
        record = {
            "timestamp": now,
            "request_id": request_id,
            "request_payload": request_payload,
            "response_payload": response_payload,
            "status_code": status_code,
            "latency_ms": latency_ms
        }
        logger.info(record)
        return True
    except Exception as e:
        logger.error(f"Erro ao logar requisição: {e}")
        return False 