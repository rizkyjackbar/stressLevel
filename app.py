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
        'Ringan': 'Kamu cuma perlu istirahat sebentar. Coba santai sejenak, dengar musik favorit, atau jalan-jalan kecil. Kalau ada waktu, kamu juga bisa lakukan hobi yang bikin kamu happy, seperti baca buku, nonton film lucu atau berkumpul sama teman. Jangan lupa minum cukup air dan makan makanan yang sehat ya. Hal kecil ini bisa bikin kamu merasa lebih segar dan siap menghadapi aktivitas lagi.',
        'Sedang': 'Stresnya lumayan nih. Coba deh meditasi, olahraga ringan, atau tarik napas dalam-dalam buat relaksasi. Ngobrol sama teman dekat atau keluarga juga bisa bantu kamu merasa lebih lega. Kalau ada waktu, coba keluar rumah dan nikmati udara segar, mungkin sambil jalan-jalan santai. Ingat, nggak apa-apa untuk berhenti sejenak dan fokus sama diri sendiri. Kamu nggak sendirian, dan ada banyak cara buat merasa lebih baik.',
        'Berat': 'Kondisinya kelihatan cukup berat. Jangan ragu buat cari bantuan profesional seperti konselor, psikolog, atau terapis. Mereka bisa bantu kamu memahami apa yang kamu rasakan dan memberikan solusi yang tepat. Selain itu, coba ngobrol sama orang yang kamu percaya, seperti keluarga atau sahabat, supaya kamu nggak merasa sendirian. Lakukan hal-hal kecil yang bikin nyaman, seperti dengar musik yang menenangkan atau menulis di jurnal tentang apa yang kamu rasakan. Ingat, nggak apa-apa untuk minta bantuan. Kamu punya hak untuk merasa lebih baik.'
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
