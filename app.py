from flask import Flask, request, jsonify
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import os
import time

app = Flask(__name__)
encoder = VoiceEncoder()

os.makedirs("uploads", exist_ok=True) 

@app.route('/', methods=['POST'])
def get_embedding():
    if 'file' not in request.files:
        return jsonify({"error": "File not found :("}), 400

    UPLOAD_FOLDER = 'uploads'
    file = request.files['file']
    timestamp = int(time.time()) 
    filename = f"{timestamp}_{file.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    
    file.save(path)

    wav = preprocess_wav(path)
    embedding = encoder.embed_utterance(wav)

    embedding_list = embedding.tolist()
    os.remove(path)

    return jsonify({"embedding": embedding_list})

if __name__ == '__main__':
    app.run(debug=True)
