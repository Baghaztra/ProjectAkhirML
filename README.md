# 📊 Prediksi Penumbang Bandara

## 📌 Deskripsi

Proyek ini adalah tugas akhir mata kuliah Machine Learning semester 5. Machine Learning untuk memprediksi jumlah penumpang bandara dengan data bersumber dari BPS. Aplikasi ini dibuat menggunakan **Flask** sebagai backend dan beberapa algoritma machine learning dari **scikit-learn** dan **XGBoost** untuk memproses data dan membuat prediksi. Model yang telah dilatih disimpan menggunakan **pickle** dan diintegrasikan ke dalam aplikasi Flask.

## 📦 Teknologi yang Digunakan

* Python 3.x
* Flask
* scikit-learn
* Pickle

## 📈 Algoritma yang Dicoba

1. **Linear Regression**
2. **Random Forest Regressor**
3. **XGBoost Regressor**

Semua model di-training dengan dataset yang sama, kemudian dibandingkan performanya menggunakan metrik **Mean Squared Error (MSE)** dan **R-squared (R²)**. Model terbaik dipilih untuk digunakan di aplikasi.

## 🛠️ Cara Menjalankan Aplikasi

1. Clone repository ini

2. Masuk ke direktori aplikasi:
   ```
   cd aplication
   ```

3. Jalankan aplikasi Flask:

   ```
   python app.py
   ```
4. Buka browser di `http://localhost:5000`

## 📊 Hasil Akhir

Setelah pengujian beberapa algoritma, didapatkan model terbaik berdasarkan nilai MSE dan R². Model tersebut disimpan menggunakan pickle dan dipakai untuk prediksi di aplikasi Flask.

## 📚 Catatan

* Dataset dapat diganti sesuai kebutuhan, pastikan formatnya sesuai.
* Jika ingin melatih ulang model, cukup jalankan `training.py` kembali.

## 🪧 Tim 

* Baghaztra Van Ril
* Aditya Ahmad Alfarison
* Vina Delta Sari

---

