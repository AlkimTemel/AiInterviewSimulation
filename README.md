🎯 AI Interview Simulation & Emotion Coach
A Real-Time Facial Emotion Recognition and Interview Coaching Platform.

📝 Overview
This project is an AI-driven interview simulation tool that analyzes a user's real-time facial expressions during a mock interview. By leveraging deep learning models, the system processes webcam feeds, detects primary emotions, and provides constructive feedback to help candidates improve their non-verbal communication skills and interview performance.

🧠 Model Engineering & Optimization Process
A core focus of this project is the iterative engineering and optimization of the underlying deep learning architecture. Rather than relying on a single attempt, the project demonstrates a structured, scientific approach to model improvement.

Both the baseline model and the optimized model are preserved in this repository for benchmarking and comparative analysis.

1. Baseline Model (emotion_model.keras)
Architecture: (Buraya ilk modelin katman yapısını kısaca yazabilirsin, örn: Basic CNN architecture with 3 Conv2D layers)

Performance: Established the initial baseline for facial feature extraction and emotion classification.

Metrics: See basarim_grafikleri.png for the initial training vs. validation accuracy and loss curves.

2. Optimized Model (emotion_model_optimized.keras)
Enhancements: Identified bottlenecks in the baseline model and introduced specific optimizations such as Dropout layers to prevent overfitting, Data Augmentation, learning rate adjustments.

Performance: Achieved higher validation accuracy and better generalization on unseen diverse facial expressions.

Metrics: See basarim_grafikleri_optimized.png for the improved convergence and stability graphs.

✨ Features
Real-Time Processing: Captures and processes video frames directly from the webcam (/process_frame endpoint).

Emotion Classification: Accurately categorizes real-time facial expressions into primary emotions using a trained Keras model.

Interactive UI: A user-friendly web interface built with HTML/CSS and served via Flask.

Comprehensive Assessment: (/analyze endpoint) Combines text and emotion analysis to provide a final holistic evaluation of the candidate's performance.

📂 Repository Structure
Plaintext
AiInterviewSimulation/
├── dataset/                             # Raw and processed data used for training
├── templates/                           # HTML files for the web interface (index.html)
├── app.py                               # Main Flask application and API endpoints
├── emotion_model.keras                  # Baseline trained model
├── basarim_grafikleri.png               # Baseline performance metrics
├── emotion_model_optimized.keras        # Final optimized model
├── basarim_grafikleri_optimized.png     # Optimized performance metrics
├── explore_data.py                      # Scripts for initial data exploration and EDA
├── train_model.py                       # Model architecture definitions and training script
├── test_models.py                       # Benchmarking and validation scripts
└── realtime_face.py                     # OpenCV integration for real-time detection
🚀 Installation & Quick Start
Prerequisites
Python 3.8+

A working webcam

Setup Instructions
1. Clone the repository:

Bash
git clone https://github.com/AlkimTemel/AiInterviewSimulation.git
cd AiInterviewSimulation
2. Install dependencies:
(Not: Eğer bir requirements.txt dosyan varsa onu belirttim, yoksa bu adımı güncelleyebilirsin)

Bash
pip install flask tensorflow opencv-python numpy
3. Run the application:

Bash
python app.py
4. Access the Interface:
Open your web browser and navigate to http://localhost:5000 to start the simulation.
