import os
import logging
from dataclasses import dataclass
import json
from typing import Union
from pathlib import Path


@dataclass
class Config:
    ROOT_PATH: str = Path(__file__).resolve().parent
    MODELS_FOLDER: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/Models"
    DATA_FOLDER: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/Data"
    MODEL: str = "model1"
    FRAME_LENGTH: int = 256
    FRAME_STEP: int = 160
    FFT_LENGTH: int = 384
    PORT: int = 8000
    HOST: str = "127.0.0.1"
    UVICORN_LOG_CONFIG_PATH: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/uvicorn_log_config.json"
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG
    LOGS_FOLDER: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/logs"

    def get_uvicorn_logger(self) -> dict:
        with open(self.UVICORN_LOG_CONFIG_PATH) as f:
            log_config = json.load(f)
            log_config["handlers"]["file_handler"]["filename"] = f"{Config.ROOT_PATH}/logs/api_logs.log"
            return log_config