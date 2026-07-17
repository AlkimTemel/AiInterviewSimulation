import google.generativeai as genai

# API anahtarını buraya yapıştır
genai.configure(api_key="YOUR API KEY")

print("Senin API Anahtarın İçin Desteklenen Modeller:\n" + "-"*40)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("Hata:", e)
