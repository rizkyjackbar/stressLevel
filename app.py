from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model dan scaler
model_path = 'model/stress_model.pkl'
scaler_path = 'model/scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("Model or scaler file not found. Please ensure the paths are correct.")

# Halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [float(data.get(var, 0)) for var in ['self_esteem', 'mental_health_history', 'depression', 'headache', 
                                                    'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 
                                                    'living_conditions', 'safety', 'basic_needs', 'academic_performance', 
                                                    'study_load', 'teacher_student_relationship', 'future_career_concerns', 
                                                    'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']]
    
    # Menambahkan nilai default untuk anxiety_level
    features.insert(0, 0)  # Nilai default untuk anxiety_level adalah 0
    
    # Mengubah input ke dalam bentuk array dan melakukan scaling
    final_features = scaler.transform([features])
    
    # Melakukan prediksi
    prediction = model.predict(final_features)
    stress_level_label = ['Ringan', 'Sedang', 'Berat']
    output = stress_level_label[prediction[0]]
    
    # Saran berdasarkan tingkat stres
    saran = {
        'Ringan': 'Lakukan aktivitas yang menyenangkan dan pastikan istirahat yang cukup.',
        'Sedang': 'Cobalah teknik relaksasi seperti meditasi atau yoga, dan bicaralah dengan teman atau keluarga.',
        'Berat': 'Pertimbangkan untuk berkonsultasi dengan profesional kesehatan mental untuk mendapatkan bantuan lebih lanjut.'
    }
    
    response = {
        'prediction_text': f'Prediksi Tingkat Stres Anda: {output}',
        'saran': saran[output]
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
