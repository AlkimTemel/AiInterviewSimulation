import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 1. Eğittiğimiz Modeli Yükleme
model = load_model('emotion_model.keras')
# Modelimizin çıktı sırası (Kaggle klasörlerinin alfabetik sırasına göredir)
class_labels = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

# 2. OpenCV'nin Hazır Yüz Bulma Algoritması (Haar Cascade)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Web Kamerasını Başlatma
cap = cv2.VideoCapture(0)

print("Kamera açılıyor... Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü gri tona çevirme (Modelimiz gri tonlama ile eğitildi)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit etme
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Yüzü kare içine alma
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Sadece yüzün olduğu bölgeyi kırpma (ROI - Region of Interest)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Kırpılan yüzü modelimizin istediği 48x48 boyutuna getirme
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Resmi Keras modelinin anlayacağı dizi formatına çevirme
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Modeli kullanarak tahmin yapma
        prediction = model.predict(roi, verbose=0)[0]
        label = class_labels[prediction.argmax()]
        
        # Tahmin edilen duyguyu yüzün üstüne yazdırma
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # Görüntüyü ekrana yansıtma
    cv2.imshow('Otonom IK - Duygu Analizi', frame)
    
    # 'q' tuşuna basıldığında döngüden çıkma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()