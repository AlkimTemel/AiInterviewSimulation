import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

# Keras ve görüntü işleme kütüphaneleri
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# --- API AYARLARI ---
os.environ["GEMINI_API_KEY"] = "YOUR API KEY"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = 'gemini-pro' # Veya sende çalışan 'gemini-1.5-flash' vb. model adı

# --- DUYGU MODELİ VE YÜZ TANIMA YÜKLEMESİ ---
try:
    emotion_model = load_model('emotion_model_optimized.keras')
    # OpenCV'nin varsayılan yüz tanıma dosyasını otomatik buluyoruz
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("✅ Duygu modeli ve yüz tanıma başarıyla yüklendi!")
except Exception as e:
    print(f"⚠️ Model yükleme hatası (Dosyanız eksik olabilir): {e}")

EMOTIONS = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

# O anki mülakattaki duyguları hafızada tutacağımız liste
session_emotions = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_question', methods=['POST'])
def generate_question():
    global session_emotions
    session_emotions = [] # Yeni mülakatta duygu geçmişini sıfırla
    data = request.json
    
    prompt = f"""Aday Profili: {data.get('egitim')}, {data.get('departman')}, {data.get('kultur')}, {data.get('soft_skill')}
    Talimat: Bu adaya TEK BİR mülakat sorusu üret. 
    KURAL: SADECE soruyu yaz, başına veya sonuna hiçbir açıklama (Soru: vb.) ekleme."""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        clean_question = response.text.strip()
        if clean_question.lower().startswith("soru:"):
            clean_question = clean_question[5:].strip()
        return jsonify({"status": "success", "question": clean_question})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- YENİ: KAMERADAN GELEN GÖRÜNTÜLERİ İŞLEME ---
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        # Tarayıcıdan gelen Base64 görseli OpenCV formatına (numpy array) çeviriyoruz
        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        current_emotion = "Yuz Bulunamadi"
        for (x, y, w, h) in faces:
            # Yüzü kırp, 48x48 boyutuna getir ve modele ver
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Modeli çalıştır ve sonucu al
            prediction = emotion_model.predict(roi, verbose=0)[0]
            label = EMOTIONS[prediction.argmax()]
            current_emotion = label
            
            # Bulunan duyguyu mülakat özeti için listeye ekle
            session_emotions.append(label)
            break # Ekranda sadece adayın (ilk bulduğu) yüzüne odaklansın

        return jsonify({"emotion": current_emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- GÜNCELLENDİ: NİHAİ DEĞERLENDİRME (METİN + DUYGU) ---
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    user_text = data.get('text', '')
    question = data.get('question', '')

    # Mülakat boyunca en çok hangi duygu yaşandıysa onu bul
    emotion_summary = "Duygu verisi toplanamadı."
    if session_emotions:
        most_common = max(set(session_emotions), key=session_emotions.count)
        emotion_summary = f"Mülakat boyunca adayın yüz ifadelerinde en yoğun gözlemlenen duygu: {most_common}"

    prompt = f"""Sen Baş İK Denetçisisin.
    Adaya sorulan soru: '{question}'
    Adayın Cevabı: '{user_text}'
    Yüz İfadeleri ve Duygu Analizi Raporu: '{emotion_summary}'

    Görevin: Adayın verdiği teknik/sözel cevabı ve mülakat anındaki duygu durumunu (örneğin stresliyse bunu yönetebilmesini, mutluysa özgüvenini) sentezleyerek ona doğrudan hitap eden profesyonel bir geri bildirim raporu yaz."""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return jsonify({"status": "success", "feedback": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
