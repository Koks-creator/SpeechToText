from typing import Tuple
import logging
from fastapi import FastAPI, UploadFile, HTTPException, Request
import tensorflow as tf
import numpy as np
from io import BytesIO
from scipy.io import wavfile
from pydantic import BaseModel
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from predictor import SpeechPredictor
from config import Config

logger = logging.getLogger("middleware_logger")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Incoming request: {request.method} {request.url}")
        try:
            response = await call_next(request)
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            response.body_iterator = iter([response_body])
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response body: {response_body.decode('utf-8')[:500]}")
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        except Exception as e:
            logger.exception(f"Unhandled error: {str(e)}")
            raise e
        

class AudioConfig:
    SAMPLE_RATE = 22050
    CHANNELS = 1
    DTYPE = tf.float32
    # For 16-bit PCM
    BIT_DEPTH = 16


class UploadResponse(BaseModel):
    transcryption: str


app = FastAPI(title="SpeechModelApi")
speech_predictor = SpeechPredictor(
    model_folder=f"{Config.MODELS_FOLDER}/{Config.MODEL}",
    frame_length=Config.FRAME_LENGTH,
    frame_step=Config.FRAME_STEP,
    fft_length=Config.FFT_LENGTH
)
app.add_middleware(LoggingMiddleware)

def validate_audio(audio_tensor: tf.Tensor, sample_rate: tf.Tensor) -> Tuple[bool, str]:
    if int(sample_rate) != AudioConfig.SAMPLE_RATE:
        return False, f"Sample rate must be {AudioConfig.SAMPLE_RATE} Hz"
    
    if audio_tensor.shape[-1] != 1:  # Should be single channel
        return False, f"Audio must be single channel"
    
    return True, ""


@app.post("/upload/", response_model=UploadResponse)
async def upload(file: UploadFile):
    try:
        content = await file.read()
        sample_rate, _ = wavfile.read(BytesIO(content))
        audio, _ = tf.audio.decode_wav(content)

        success, msg = validate_audio(audio_tensor=audio, sample_rate=sample_rate)
        if success:
            tensored_audio = speech_predictor.encode_single_sample(wav_data=content)

            data = np.array([tensored_audio])
            res = speech_predictor.make_prediction(data=data)[0]

            return UploadResponse(transcryption=res)
        else:
            raise HTTPException(status_code=422, detail=msg)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/audio_specs/")
async def get_audio_specifications():
    return {
        "sample_rate": AudioConfig.SAMPLE_RATE,
        "channels": AudioConfig.CHANNELS,
        "bit_depth": AudioConfig.BIT_DEPTH,
        "format": "PCM WAV",
        "dtype": str(AudioConfig.DTYPE)
    }


@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "WAV to Tensor Processor API is running",
        "audio_specs": {
            "sample_rate": AudioConfig.SAMPLE_RATE,
            "format": "16-bit PCM WAV",
            "channels": "mono"
        }
    }


if __name__ == '__main__':
    # log_config=uvicorn_log_config
    #  log_config=Config().get_uvicorn_logger()
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_config=Config().get_uvicorn_logger())
