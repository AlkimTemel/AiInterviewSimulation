import os
import matplotlib.pyplot as plt
import cv2

# Eğitim klasörünün yolu
dataset_path = 'dataset/train/'

# Eğer klasör bulunamazsa uyar
if not os.path.exists(dataset_path):
    print(f"Hata: '{dataset_path}' klasörü bulunamadı. Lütfen ZIP dosyasını doğru çıkardığından emin ol.")
else:
    emotions = os.listdir(dataset_path)
    plt.figure(figsize=(12, 6))

    # Her duygudan birer örnek resim alıp ekrana çizdirelim
    for i, emotion in enumerate(emotions):
        folder_path = os.path.join(dataset_path, emotion)
        
        # Klasör içindeki ilk resmin adını al
        img_name = os.listdir(folder_path)[0]
        img_path = os.path.join(folder_path, img_name)
        
        # Resmi OpenCV ile siyah-beyaz (grayscale) oku
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        plt.subplot(2, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(emotion.upper(), fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("Veri seti başarıyla yüklendi ve örneklendi!")