from flask import Flask, request, jsonify
from resemblyzer import preprocess_wav, VoiceEncoder
from pydub import AudioSegment
import os
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
encoder = VoiceEncoder()

UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)
AudioSegment.converter = "ffmpeg"

@app.after_request
def add_cors_headers(response):
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"  
    response.headers["Access-Control-Allow-Origin"] = "*"  
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route('/', methods=['POST'])
def get_embedding():
    if 'file' not in request.files:
        return jsonify({"error": "File not found :("}), 400

    file = request.files['file']
    timestamp = int(time.time())

    webm_filename = f"{timestamp}_{file.filename}"
    webm_path = os.path.join(UPLOAD_FOLDER, webm_filename)
    file.save(webm_path)

    mp3_filename = webm_filename.replace('.webm', '.mp3')
    mp3_path = os.path.join(CONVERTED_FOLDER, mp3_filename)

    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(mp3_path, format="mp3")

        wav = preprocess_wav(mp3_path)
        embedding = encoder.embed_utterance(wav)
        embedding_list = embedding.tolist()

    except Exception as e:
        return jsonify({"error": f"Conversion or embedding error: {str(e)}"}), 500

    return jsonify({
        "message": "Embedding extracted successfully",
        "embedding": embedding_list
    })

if __name__ == '__main__':
    app.run(debug=True)
