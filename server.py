from flask import Flask, request, jsonify, send_file
import torch
import io
import soundfile as sf
from f5_tts import api  # Assuming f5_tts provides an API class for TTS inference
import asyncio
import concurrent.futures
import os
import whisper
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import subprocess
import torch
from concurrent.futures import ThreadPoolExecutor


app = Flask(__name__)

class TTSModel:
    def __init__(self, model_name="F5-TTS", device="cuda"):
        self.device = device
        # Assuming F5TTS internally handles device placement via the 'device' argument
        self.tts_model = api.F5TTS(device=device)  # Initialize the TTS model
        print(f"{model_name} model loaded successfully on {device}")

        # Ensure CUDA is available and being utilized
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")
        
        # You may need to modify this further if 'F5TTS' has its own device management
        # If it does not support the '.to()' method, check the F5TTS documentation for the correct device handling.

    def infer(self, ref_audio_path, ref_text, gen_text):
        with torch.no_grad():  # Disable gradient calculation for inference
            # Directly pass gen_text as a string, without converting to a tensor
            wav, _, _ = self.tts_model.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=gen_text,
                seed=-1
            )

            # Write output to a buffer
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, wav, samplerate=22050, format="WAV")
            audio_buffer.seek(0)
            return audio_buffer


device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTSModel(device=device)

# Set up the thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

async def process_inference(ref_audio_path, ref_text, gen_text):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, lambda: tts_model.infer(ref_audio_path, ref_text, gen_text))
    return result

@app.route('/tts', methods=['POST'])
async def tts():
    data = request.get_json()
    gen_text = data.get("text", "")
    ref_text = data.get("ref_text", "I am proud India is pacing forward in addressing climate change by hosting its first ever Formula E-Race.")
    ref_audio = "8608517287293824917448_audio.mp3"  # Path to reference audio

    if not gen_text or not ref_audio:
        return jsonify({"error": "Text and reference audio are required"}), 400

    try:
        # Perform TTS inference asynchronously
        audio_buffer = await process_inference(ref_audio_path=ref_audio, ref_text=ref_text, gen_text=gen_text)
        
        # Send audio file as a response
        return send_file(audio_buffer, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Define the path to store the model and uploads
MODEL_PATH = "whisper_model"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_audio(input_path, output_path):
    command = ['ffmpeg', '-y', '-i', input_path, '-ar', '16000', '-ac', '1', output_path]
    subprocess.run(command, check=True)


# Load the Whisper model (will download only if not cached)
def load_model():
    os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure model cache directory exists

    if not os.path.exists(MODEL_PATH):
        print("Downloading Whisper model...")
    else:
        print("Using cached Whisper model...")
    
    # Use "cuda" if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")

    model = whisper.load_model("base", download_root=MODEL_PATH, device=device)
    return model


print("Loading Whisper model...")
model = load_model()


@app.route('/new_talking', methods=['POST'])
def new_talking():
    print("Received new audio file...")
    file = request.files['file']

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess the audio (convert to 16kHz mono)
    preprocessed_path = os.path.join(UPLOAD_FOLDER, f"processed_{filename}")
    preprocess_audio(filepath, preprocessed_path)

    # Perform transcription
    result = model.transcribe(preprocessed_path)
    print("Transcription complete.")

    return result

async def process_inference(ref_audio_path, ref_text, gen_text):
    loop = asyncio.get_event_loop()
    audio_buffer = await loop.run_in_executor(executor, tts_model.infer, ref_audio_path, ref_text, gen_text)
    return audio_buffer


if __name__ == "__main__":
    # Run Flask app with multi-threading enabled for better concurrency
    app.run(host="0.0.0.0", port=8080, threaded=True)
