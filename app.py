from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
import joblib  # Untuk memuat scaler
from sklearn.preprocessing import StandardScaler

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Path model dan scaler
model_path = 'model/stress_level_model_v2.h5'  # Ganti dengan path model .h5
scaler_path = 'model/scaler.pkl'

# Memuat model TensorFlow dan scaler
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = tf.keras.models.load_model(model_path)  # Muat model TensorFlow
    scaler = joblib.load(scaler_path)  # Memuat scaler
    print("Model dan scaler berhasil dimuat.")
else:
    raise FileNotFoundError("Model atau scaler tidak ditemukan. Pastikan path sudah benar.")

# Halaman utama
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data dari request JSON
    data = request.json
    
    # Memastikan input data sesuai dengan urutan fitur yang diperlukan oleh model
    features = [
        float(data.get(var, 0)) for var in [
            'self_esteem', 'mental_health_history', 'depression', 'headache', 
            'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 
            'living_conditions', 'safety', 'basic_needs', 'academic_performance', 
            'study_load', 'teacher_student_relationship', 'future_career_concerns', 
            'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying'
        ]
    ]
    
    # Menambahkan nilai default untuk anxiety_level
    features.insert(0, 0)  # Nilai default untuk anxiety_level adalah 0
    
    # Melakukan scaling input data sesuai dengan scaler yang sudah dilatih
    final_features = scaler.transform([features])
    
    # Melakukan prediksi dengan model TensorFlow
    prediction = model.predict(final_features)
    
    # Mengambil kelas prediksi dengan probabilitas tertinggi
    predicted_class = np.argmax(prediction, axis=1)
    
    # Label untuk tingkat stres
    stress_level_label = ['Ringan', 'Sedang', 'Berat']
    output = stress_level_label[predicted_class[0]]

    # Menentukan emoticon berdasarkan tingkat stres
    emoticons = {
        'Ringan': '😄',
        'Sedang': '😐',
        'Berat': '😢'
    }
    emoticon = emoticons[output]

    # Saran berdasarkan tingkat stres
    saran = {
        'Ringan': 'Lakukan aktivitas yang menyenangkan dan pastikan istirahat yang cukup.',
        'Sedang': 'Cobalah teknik relaksasi seperti meditasi atau yoga, dan bicaralah dengan teman atau keluarga.',
        'Berat': 'Pertimbangkan untuk berkonsultasi dengan profesional kesehatan mental untuk mendapatkan bantuan lebih lanjut.'
    }
    
    # Mempersiapkan response dalam format JSON
    response = {
        'prediction_text': f'Prediksi Tingkat Stres Anda: {output}',
        'saran': saran[output],
        'emoticon': emoticon
    }
    
    return jsonify(response)

# Menjalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
