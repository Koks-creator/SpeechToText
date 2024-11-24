from dataclasses import dataclass
import numpy as np
import pickle
from pathlib import Path
from typing import Union, List
import os
from tensorflow.types.experimental import TensorLike
import tensorflow as tf
from tensorflow import keras

from config import Config
from custom_logger import CustomLogger

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=fr"{Config.LOGS_FOLDER}/predictor_logs.log"
).create_logger()


def CTCLoss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def ctc_accuracy(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    batch_size = tf.shape(y_pred)[0]
    max_length = tf.shape(y_pred)[1]
    input_lengths = tf.fill([batch_size], max_length)

    # Convert y_true to SparseTensor
    y_true = tf.cast(y_true, tf.int64)
    indices = tf.where(tf.not_equal(y_true, 0))
    values = tf.gather_nd(y_true, indices)
    dense_shape = tf.cast(tf.shape(y_true), tf.int64)
    y_true = tf.SparseTensor(indices, values, dense_shape)

    # Decode the predictions
    decoded, _ = tf.nn.ctc_greedy_decoder(
        tf.transpose(y_pred, [1, 0, 2]),
        input_lengths,
        merge_repeated=True
    )

    decoded_sparse = tf.cast(decoded[0], tf.int64)

    # Calculate accuracy using edit_distance
    accuracy = 1 - tf.reduce_mean(
        tf.edit_distance(decoded_sparse, y_true, normalize=True)
    )

    return accuracy
    

@dataclass
class SpeechPredictor:
    model_folder: Union[str, os.PathLike]
    frame_length: int = 256
    frame_step: int = 160
    fft_length: int = 384

    def __post_init__(self) -> None:
        logger.info("Initing predictor with: \n"
                    f"{self.model_folder=} \n"
                    f"{self.frame_length=} \n"
                    f"{self.frame_step=} \n"
                    f"{self.fft_length=} \n"
                    )
        try:
            self.model_path = list(Path(self.model_folder).glob("*.h5"))[0].absolute()
            logger.info(f"{self.model_path=}")
            self.model = keras.models.load_model(self.model_path,
                                                custom_objects={"CTCLoss": CTCLoss,
                                                                "ctc_accuracy": ctc_accuracy
                                                                })
            
            n2c_path = Path(f"{self.model_folder}/num_to_char.pkl").absolute()
            with open(n2c_path, "rb") as n2c_f:
                self.num_to_char = pickle.load(n2c_f)
            
            c2n_path = Path(f"{self.model_folder}/char_to_num.pkl").absolute()
            with open(c2n_path, "rb") as c2n_f:
                self.char_to_num = pickle.load(c2n_f)
            
            logger.info("Model and char mappings have been loaded")
        except (FileExistsError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Could not load a model or vocabulary mappings (or both): {e}")
            raise Exception(f"Could not load a model or vocabulary mappings (or both): {e}")
        except IndexError as e:
            logger.error(f"Model file not found: {e}")
            raise Exception(f"Model file not found: {e}")
        except Exception as e:
            logger.error(f"Unknown error prevented model loading: {e}")
            raise Exception(f"Unknown error prevented model loading: {e}")
    
    def encode_single_sample(self, wav_data: Union[str, os.PathLike, Path, TensorLike]) -> TensorLike:
        if isinstance(wav_data, str):
            wav_data = tf.io.read_file(wav_data)

        audio, _ = tf.audio.decode_wav(wav_data)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)

        # Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length
        )
        # Only magnitude is needed, which can be get by using tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, .5) # applies a square root, which can make differences in the magnitude values less extreme.
                                                   # This is done for normalization, so that very high values don't dominate the spectrogram.

        # Normalize the spectrogram
        # Here, each column of the spectrogram is standardized by subtracting its mean and dividing by its standard deviation.
        # This ensures that all values have a mean close to 0 and a standard deviation close to 1,
        # making it easier for neural networks to process without bias toward high-magnitude frequency components.
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        return spectrogram

    def decode_batch_predictions(self, pred) -> List[str]:
        # pred has shape of (batch, time_steps, and len of vocubalary)
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

        output_text = []
        for result in results:
            # concatenates the characters into a single string in utf-8
            result = tf.strings.reduce_join(self.num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text
    
    def make_prediction(self, data: np.array) -> List[str]:
        try:
            batch_predictions = self.model.predict(data)
            batch_predictions = self.decode_batch_predictions(batch_predictions)
            return batch_predictions
        except Exception as e: # yes, turbo broad exception, and what u gonna do :)?
            logger.error(f"Unhandled error: {e}", exc_info=True)
            raise e

if __name__ == "__main__":
    speech_predictor = SpeechPredictor(
        model_folder=f"{Config.MODELS_FOLDER}/{Config.MODEL}",
        frame_length=Config.FRAME_LENGTH,
        frame_step=Config.FRAME_STEP,
        fft_length=Config.FFT_LENGTH
    )
    d = tf.io.read_file(f"{Config.DATA_FOLDER}\LJ025-0076.wav")
    x = speech_predictor.encode_single_sample(wav_data=d)
    data = np.array([x])
    print(speech_predictor.make_prediction(data=data))
