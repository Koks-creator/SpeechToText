import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from difflib import SequenceMatcher

from api import app
from config import Config


@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200

def test_audio_specs_endpoint(client):
    response = client.get("/audio_specs")
    assert response.status_code == 200

def test_correct_wav(client):
    with open(fr"{Config.ROOT_PATH}/tests/LJ009-0025.wav", "rb") as f:
        files = {"file": ("test_audio.wav", f, "audio/wav")}
        response = client.post("/upload/", files=files)
        assert response.status_code == 200
        data = response.json()
        assert SequenceMatcher(
            None, data["transcryption"],
            "as to the exclusion of strangers on these occasions"
            ).ratio() > .9

def test_uncorrect_wav(client):
    with open(fr"{Config.ROOT_PATH}/tests/left1.wav", "rb") as f:
        files = {"file": ("test_audio.wav", f, "audio/wav")}
        response = client.post("/upload/", files=files)
        print(response.content)
        assert response.status_code == 422