import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. Veri Yolları
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# 2. Veri Çoğaltma (Daha agresif bir augmentation kullanıyoruz ki model ezberlemesin)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# 3. OPTİMİZE EDİLMİŞ DERİN CNN MİMARİSİ
model = Sequential([
    # 1. Evrişim Bloğu
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 2. Evrişim Bloğu
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 3. Evrişim Bloğu
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 4. Evrişim Bloğu (Yeni eklendi)
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Sınıflandırma Katmanları
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Özel Optimizer (Başlangıç öğrenme hızını ayarlıyoruz)
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Gelişmiş Callbacks
# Model eski dosyanın üzerine yazmasın diye adını değiştirdik, ikisini de raporda kıyaslarız!
checkpoint = ModelCheckpoint('emotion_model_optimized.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Yeni: Öğrenme oranını dinamik düşürme
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

# 5. Modeli Eğitme (Epok sayısını 50'ye çıkardık)
print("Optimize edilmiş model eğitimi başlıyor... (Bu işlem önceki modele göre daha uzun sürecektir)")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# 6. Yeni Başarım Grafikleri
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.title('Optimize Model Kayıp (Loss) Grafiği')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
plt.title('Optimize Model Başarım (Accuracy) Grafiği')
plt.legend()

plt.savefig('basarim_grafikleri_optimized.png')
plt.show()

print("Eğitim tamamlandı! En iyi model 'emotion_model_optimized.keras' olarak kaydedildi.") 