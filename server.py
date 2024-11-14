from flask import Flask, request, jsonify, send_file
import torch
import io
import soundfile as sf
import asyncio
from f5_tts import api  # Assuming f5_tts provides an API class for TTS inference

app = Flask(__name__)

class TTSModel:
    def __init__(self, model_name="F5-TTS", device="cpu"):
        self.device = device
        self.tts_model = api.F5TTS(device=device)  # Initialize the TTS model
        print(f"{model_name} model loaded successfully on {device}")
        
        # Set flag for using mixed precision only during inference
        self.use_fp16 = device == "cuda" and torch.cuda.is_available()

        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")

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


# Instantiate TTS model on GPU if available
tts_model = TTSModel(device="cuda" if torch.cuda.is_available() else "cpu")

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    gen_text = data.get("text", "")
    ref_text = data.get("ref_text", "I am proud India is pacing forward in addressing climate change by hosting its first ever Formula E-Race.")
    ref_audio = "8608517287293824917448_audio.mp3"  # Path to reference audio

    if not gen_text or not ref_audio:
        return jsonify({"error": "Text and reference audio are required"}), 400

    try:
        # Perform TTS inference synchronously
        audio_buffer = tts_model.infer(ref_audio_path=ref_audio, ref_text=ref_text, gen_text=gen_text)
        
        # Send audio file as a response
        return send_file(audio_buffer, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use multiple threads for improved concurrency
    app.run(host="0.0.0.0", port=8000, threaded=True)
