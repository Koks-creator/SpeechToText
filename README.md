# Speech-to-Text API
A FastAPI-based REST API for transcribing WAV audio files to text using a TensorFlow model. The API processes audio through spectrogram generation and uses CTC (Connectionist Temporal Classification) for text prediction.

[![video](https://img.youtube.com/vi/leUCpi6dmJk/0.jpg)](https://www.youtube.com/watch?v=leUCpi6dmJk)

## 📋 Requirements
 * fastapi==0.115.5
 * numpy==1.26.0
 * pydantic==2.5.3
 * pytest==8.3.3
 * Requests==2.32.3
 * scipy==1.14.1
 * starlette==0.41.2
 * tensorflow==2.17.0
 * tensorflow_intel==2.17.0
 * uvicorn==0.32.0

## 🗂️ Project Structure
```plaintext
├── logs/ # create this folder
├── Models/
│   └── model1/      
│       ├── model.h5
│       ├── num_to_char.pkl
│       └── char_to_num.pkl
├── Data/
├── api.py              
├── config.py           
├── custom_logger.py    
├── predictor.py       
└── tests/
    └── test_app.py   
```

## 🚀 Setup
### Folders
Create 'logs' folder

### Dependencies Installation
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Key Settings (`config.py`)
* Frame Length: `256`
* Frame Step: `160`
* FFT Length: `384`
* Default Port: `8000`
* Default Host: `127.0.0.1`

### Audio Specifications
* Sample Rate: **22050 Hz**
* Channels: **1 (mono)**
* Format: **16-bit PCM WAV**
* Data Type: **float32**

## 🔌 API Endpoints

### `GET /`
Health check endpoint returning API status and audio requirements.

**Response:**
```json
{
    "status": "ok",
    "message": "WAV to Tensor Processor API is running",
    "audio_specs": {
        "sample_rate": 22050,
        "format": "16-bit PCM WAV",
        "channels": "mono"
    }
}
```

### `GET /audio_specs/`
Returns detailed audio specifications for input files.

### `POST /upload/`
Transcribes WAV audio file to text.

**Request:**
* Method: `POST`
* Content-Type: `multipart/form-data`
* Body: WAV file

**Response:**
```json
{
    "transcryption": "transcribed text"
}
```

## 📝 Logging

**The application implements comprehensive logging:**
* Request/response logging via middleware
* Predictor operations logging
* Error tracking
* Separate log files for API and predictor

## ⚠️ Error Handling

**The API includes validation for:**
* Audio sample rate (must be 22050 Hz)
* Channel count (must be mono)
* File format compatibility
* Model prediction errors

## 🏃‍♂️ Running the API

**Start the server:**
```bash
python api.py
```

> The API will be available at `http://127.0.0.1:8000`

## 🎙️ More about predictor
**predictor.py** file takes path to wav file or loaded audio as as input, convert audio to the spectogram using **encode_single_sample** method, then spectogram is passed to the loaded speech to text model and finally prediction is being decoded with **decode_batch_predictions** method.

## 🧪 Testing

**Run the test suite:**
```bash
pytest tests/test_app.py
```

**Tests cover:**
* Basic endpoint functionality
* Valid WAV file processing
* Invalid WAV file handling
* Audio specification validation

## 🗣️ About Dataset
For this project I used the LJ Speech Dataset which is: 
<br>
>"public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours."
>
This dataset contains recording of only 1 female speaker, so don't expect models trained on this dataset to perform well on your voice for example

## 📓📚 Training model
You can train model using **SpeechToTextModel.ipynb**
<br>
Before you start you need to create folder on you google drive for checkpoints - checkpoints are really important in this case since one epoch on GPU can take around 1hr (first epoch) next epochs train for around 20 minutes - don't even try training this model on CPU, if you want to load checpoint just set LOAD_CHECKPOINT to true, training this model for around 20+ epochs is fine.
<br>
<br>
Another thing to keep your eye on is training your model on tensorflow version that will be compatible with tensorflow version on your machine or wherever you wanna run it. 
<br>
<br>
To run this model you don't need GPU (only for training is 100% required).

## 📊🎯 Training results
![res](https://github.com/user-attachments/assets/3afedfc4-6d30-4bfe-8d53-c22f852dbecc)


## 🧠  DeepSpeech-Inspired Speech Recognition Model

This model is inspired by DeepSpeech2, designed for speech-to-text applications. It processes audio input (spectrograms) through convolutional and recurrent layers to generate textual predictions.

### Model Architecture

#### 1. Model Input
* **Shape**: The model accepts a 2D spectrogram as input, represented as `(time_steps, frequency_bins)`, where each time step corresponds to a time frame, and each frequency bin represents spectral information at that frame.
* **Layer**: `input_spectrogram` serves as the input layer, handling spectrograms with variable time steps (`None` in the time dimension).

#### 2. Expand Dimensions for 2D Convolutional Layers
* **Reshape Layer**: `layers.Reshape` expands the input to `(time_steps, frequency_bins, 1)` to allow 2D convolutional processing.

#### 3. Convolutional Layers
The model applies two 2D convolutional layers to extract features from the spectrogram.
* **Conv Layer 1**:
   * Filters: 32
   * Kernel Size: `[11, 41]`
   * Strides: `[2, 2]` (reduces the input size in both time and frequency dimensions)
   * Activation: `ReLU`
   * Normalization: Batch normalization applied after convolution.
* **Conv Layer 2**:
   * Filters: 32
   * Kernel Size: `[11, 21]`
   * Strides: `[1, 2]`
   * Activation: `ReLU`
   * Normalization: Batch normalization applied after convolution.

After the convolutional layers, the output is reshaped to a sequence format to feed into the recurrent layers. Convolution layers extract features which helps with ignoring irrelevant patterns or noise in the spectrogram, so model can focus on speech.

#### 4. Bidirectional GRU Layers
The model uses stacked bidirectional GRU (Gated Recurrent Unit) layers to capture temporal dependencies in the audio sequence.
* **GRU Layers**:
   * Number of Layers: `rnn_layers` (specified as a hyperparameter)
   * Units per Layer: `rnn_units`
   * Direction: Bidirectional (processes the sequence forwards and backwards)
   * Dropout: Applied between GRU layers to prevent overfitting
* **Sigmoid Activation in GRU**: Each GRU layer uses a **sigmoid activation** function within its *update* and *reset gates*:
   * **Update Gate**: Controls the flow of information from the previous time step.
   * **Reset Gate**: Controls how much past information to "forget."
   * The sigmoid activation constrains the gate outputs to a range between 0 and 1, enabling fine control over memory retention and flow through the sequence. This makes GRUs effective at handling longer-term dependencies in sequences.

#### 5. Dense Layer
* A fully connected layer (`dense_1`) is used to further process the output of the recurrent layers.
   * Units: `rnn_units * 2`
   * Activation: `ReLU`
   * Dropout: Applied after this layer for additional regularization.

#### 6. Output Layer
* The final output layer is a fully connected layer with `output_dim + 1` units, where each unit represents a character or a blank symbol (for CTC loss).
* **Activation**: `softmax`, which converts the output to a probability distribution over possible characters.


## 🔍 CTC loss
CTC loss, or **Connectionist Temporal Classification Loss**, is a loss function used primarily in sequence recognition tasks where the output length is not specified. It is particularly useful in speech recognition, image-to-text transcription (OCR) and other sequence processing model

 - **How does CTC Loss work?** <br>
Independence of input and output length: In many cases, the length of the input sequence (e.g., the number of frames in an audio recording) differs from the length of the expected output sequence (e.g., the number of characters in a transcription). CTC is designed to handle this problem.

- **Inserting a “blank” symbol**: <br>
 CTC uses an additional blank symbol (meaning “no character”) to help deal with the situation when the model does not predict a new character in each input frame. Blank symbols allow the model to differentiate between adjacent characters and ignore redundant predictions.

- **Different matches (alignment)**: <br>
CTC calculates the probability of different possible matches between the input sequence and the output sequence. For example, for an input audio signal that corresponds to the output “HELLO,” the model can generate multiple combinations (e.g., “H_E_LL_O_” where “_” is a blank) that will be interpreted as correct transcriptions.

In sequential tasks like speech recognition, the length of the input (e.g., the number of audio frames) is variable, as is the length of the output (e.g., the number of words or characters in a transcription). CTCLoss is designed specifically to deal with this variability in length, allowing for different input to output matches and inserting blank symbols in the appropriate place

## 📚 Source
 * https://voidful.medium.com/understanding-ctc-loss-for-speech-recognition-a16a3ef4da92
 * https://medium.com/@akp83540/connectionist-temporal-classification-ctc-722bbb767e62
 * https://www.youtube.com/watch?v=qKz_lmgad3o
