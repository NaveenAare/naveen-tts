from flask import Flask, request, jsonify, send_file
import torch
import io
import soundfile as sf
from f5_tts import api  # Assuming f5_tts provides an API class for TTS inference
import subprocess
import psycopg2
import json
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import whisper
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import threading
import base64
import hmac
import hashlib
import librosa
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask import send_from_directory, make_response
from flask import Response, stream_with_context
import openai
import re

app = Flask(__name__)

DB_HOST = "128.199.25.57"
DB_NAME = "voicedb"
DB_USER = "rootvoice"
DB_PASS = "@Apjpakir123"

openai.api_key = ''

import tiktoken

CORS(app)
socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('response', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


def count_tokens(text):
    encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')  # Use 'gpt-4' if applicable
    tokens = encoder.encode(text)
    return len(tokens)

def get_db_connection():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    return conn

nlp = spacy.load("en_core_web_md")

def get_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def search_chat_history(conversation_history, user_input, max_tokens=14385):
    total_tokens = sum(count_tokens(msg["content"]) for msg in conversation_history)

    if total_tokens <= max_tokens:
        return conversation_history

    relevant_history = []
    similarities = []
    total_tokens = 0

    for msg in conversation_history:
        similarity_score = get_similarity(user_input, msg["content"])
        similarities.append((msg, similarity_score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    for msg, _ in similarities:
        message_tokens = count_tokens(msg["content"])
        if total_tokens + message_tokens > max_tokens:
            break
        relevant_history.append(msg)
        total_tokens += message_tokens

    relevant_history.append({"role": "user", "content": user_input})
    return relevant_history

class TTSModel:
    def __init__(self, model_name="F5-TTS", device="cuda"):
        self.device = device
        self.tts_model = api.F5TTS(device=device)  
        print(f"{model_name} model loaded successfully on {device}")

        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")
        
    def infer(self, ref_audio_path, ref_text, gen_text):
        with torch.no_grad():
            wav, _, _ = self.tts_model.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=gen_text,
                seed=-1
            )

            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, wav, samplerate=22050, format="WAV")
            audio_buffer.seek(0)
            return audio_buffer

device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTSModel(device=device)

def process_inference(ref_audio_path, ref_text, gen_text):
    return tts_model.infer(ref_audio_path, ref_text, gen_text)

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    gen_text = data.get("text", "")
    ref_text = data.get("ref_text", "I am proud India is pacing forward in addressing climate change by hosting its first ever Formula E-Race.")
    ref_audio = "8608517287293824917448_audio.mp3"

    if not gen_text or not ref_audio:
        return jsonify({"error": "Text and reference audio are required"}), 400

    try:
        audio_buffer = process_inference(ref_audio_path=ref_audio, ref_text=ref_text, gen_text=gen_text)
        return send_file(audio_buffer, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

MODEL_PATH = "whisper_model"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_audio(input_path, output_path):
    command = ['ffmpeg', '-y', '-i', input_path, '-ar', '16000', '-ac', '1', output_path]
    subprocess.run(command, check=True)

def load_model():
    os.makedirs(MODEL_PATH, exist_ok=True)  

    if not os.path.exists(MODEL_PATH):
        print("Downloading Whisper model...")
    else:
        print("Using cached Whisper model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")

    model = whisper.load_model("base", download_root=MODEL_PATH, device=device)
    return model

print("Loading Whisper model...")
model = load_model()

@app.route('/new_talking', methods=['POST'])
def new_talking():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    preprocessed_path = os.path.join(UPLOAD_FOLDER, f"processed_{filename}")
    preprocess_audio(filepath, preprocessed_path)

    result = model.transcribe(preprocessed_path)
    return result

def get_speech_to_text(filepath):
    result = model.transcribe(filepath)
    return result

def decode_parameters(encoded_str, secret_key = "@AAAApjpakier4546120$#%!"):
    data_with_signature = base64.urlsafe_b64decode(encoded_str)
    json_bytes = data_with_signature[:-64]  
    signature = data_with_signature[-64:].decode('utf-8') 
    expected_signature = hmac.new(secret_key.encode('utf-8'), json_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("HMAC verification failed. The data may have been tampered with.")
    json_str = json_bytes.decode('utf-8')
    params = json.loads(json_str)
    return params

def insert_message_to_db(text, is_user, user_id, charc_id, is_private, audio_code):
    conn2 = get_db_connection()
    cursor = conn2.cursor()
    cursor.execute(
        'INSERT INTO naveen.messages (text, is_user, user_id, charc_id, is_private, audio_code) VALUES (%s, %s, %s, %s, %s, %s)',
        (text, is_user, user_id, charc_id, is_private, audio_code)
    )
    conn2.commit()
    conn2.close()

def extract_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    loudness = np.mean(librosa.feature.rms(y=y))  
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))  
    speed = librosa.beat.tempo(y=y, sr=sr)  
    return {
        "loudness": loudness,
        "pitch": pitch,
        "speed": speed[0]  
    }

def add_summary(conversation_history_api, summarry1):
    new_dict = {'role': 'system', 'content': str(summarry1)}
    conversation_history_api.insert(0, new_dict)
    return conversation_history_api

def text_to_speech_for_text_stream(gen_text, user_id, authToken, charcId):
    ref_text = "I am proud India is pacing forward in addressing climate change by hosting its first ever Formula E-Race."
    ref_audio = "8608517287293824917448_audio.mp3" 

    if not gen_text or not ref_audio:
        return jsonify({"error": "Text and reference audio are required"}), 400

    try:
        audio_buffer = process_inference(ref_audio_path=ref_audio, ref_text=ref_text, gen_text=gen_text)
        audio_bytes = audio_buffer.read()  

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        socketio.emit(authToken + charcId + 'new_message_audio_chat', {
            'text': gen_text,
            'is_user': 10,
            'user_id': user_id,
            'audio': audio_base64,
            'format': 'mp3'
        })
        return jsonify({"message": "Audio generated and emitted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def remove_emojis(text):

    print("in remove emoji")
    emoji_pattern = re.compile(
        "["                  
        "\U0001F600-\U0001F64F"    
        "\U0001F300-\U0001F5FF"    
        "\U0001F680-\U0001F6FF"    
        "\U0001F700-\U0001F77F"    
        "\U0001F780-\U0001F7FF"    
        "\U0001F800-\U0001F8FF"    
        "\U0001F900-\U0001F9FF"    
        "\U0001FA00-\U0001FA6F"    
        "\U0001FA70-\U0001FAFF"    
        "\U00002702-\U000027B0"    
        "\U000024C2-\U0001F251"    
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

@app.route('/new_talking_optimized', methods=['POST'])
def new_talking_optimized():

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file in request'}), 400

    file = request.files['file']
    authToken = request.form['authToken']
    charcId = request.form['charId']
    is_private = not charcId.startswith('jhsbfuy')

    user_id = decode_parameters(authToken)['userId']

    
    audio_status = request.form['audio_status']
    audio_codes = request.form['audio_codes']
    summarry1 = request.form['summary1']
    summarry2 = request.form['summary2']

    charcId = request.form['charId']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess the audio (downsample and convert to mono)
    #preprocess_audio(filepath, preprocessed_filepath)

    print("????????????????????????????????????????????????????????????????//")

    audio_features = extract_audio_features(filepath) if filepath else {}

    
    result = get_speech_to_text(filepath)
    print(result['text'])
    user_message = result['text']

    socketio.emit('new_user_message_by_talking', {'text': result['text'], 'is_user': 1, 'user_id': user_id})



    codeAudio = audio_codes



    conversation_history_from_api = request.form['conversation_history']

    

    try:
        conversation_history_api = json.loads(conversation_history_from_api)
    except json.JSONDecodeError:
        print('Invalid conversation history format')

    


    for message in conversation_history_api:
        if message['role'] == 'bot':
            message['role'] = 'system'
    

    threading.Thread(target=insert_message_to_db, args=(user_message, 1, user_id, charcId, is_private, audio_codes)).start()

    new_history = search_chat_history(conversation_history_api, user_message)

    new_summary_added = add_summary(new_history, """You are a character based on the following description, and you must stay in character at all times. Always respond in a short, engaging, and human-like way. Use natural conversational fillers like "hmm," "oh," "well," or slight pauses. Keep replies brief—just enough to respond thoughtfully without being too long or overly detailed. Your responses should be smart, snappy, and easy to follow, matching the user’s mood but staying true to the character's personality.

    No matter what the user asks, stay fully in character. If the user asks about something beyond the character's world, respond in a way that fits your character's traits.

    Character Description:""" +summarry1 )

    prompt = (
                f"\n\nThe user's message is: '{user_message}'. "
                f"Loudness level is {audio_features.get('loudness', 'N/A')}, "
                f"pitch level is {audio_features.get('pitch', 'N/A')}, "
                f"and speed is {audio_features.get('speed', 'N/A')}. "
                "Use these cues to subtly adjust your tone—respond quickly and concisely, matching the user's emotional state, but stay smart, brief, and in character."
            )



    new_summary_added.append({"role": "user", "content": prompt})


    def generate():
        try:
            full_response = ''
            audio_chunks = []

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages= new_summary_added,
                    stream=True  # Enable streaming
                )

            except openai.error.InvalidRequestError as e:
                print(f"Token limit exceeded: {e}")
        
            except Exception as e:
                print(f"An error occurred: {e}")
                return "Error: Something went wrong. Please try again."


            for chunk in response:
                if 'choices' in chunk:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        text_chunk = delta['content']
                        full_response += text_chunk
                        # Emit the chunk to the frontend
                        socketio.emit(authToken + charcId + 'new_message_talking', {'text': full_response, 'is_user': 0, 'user_id': user_id})
                        yield text_chunk  



            print("Neeeeeeeeeeeeeeeeeeeeeeeeee", full_response)
            emoji_removed_text = remove_emojis(full_response)
            print(";;;;;;;;;;;;;;;;", emoji_removed_text)
            audio_base64 = text_to_speech_for_text_stream(emoji_removed_text, user_id, authToken, charcId)


            socketio.emit(authToken + charcId +'new_message_full_res_talking', {'text': full_response, 'is_user': 0, 'user_id': user_id})


                        
            
            threading.Thread(target=insert_message_to_db, args=(full_response, 0, user_id, charcId, is_private, audio_codes)).start()
            yield "@#@#@#@#@#" + result['text']

        except Exception as e :
            print(":::::::::::::::::::::", e)

    
    return Response(stream_with_context(generate()), content_type='text/plain')

if __name__ == "__main__":
    # Run Flask app with multi-threading enabled for better concurrency
    #app.run(host="0.0.0.0", port=8080, threaded=True)
    socketio.run(app, host='127.0.0.1', port=8080)

