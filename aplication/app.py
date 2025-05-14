from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model dan scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Daftar target
targets = [
    'Hasanudin-Makassar',
    'Juanda-Surabaya',
    'Kualanamu-Medan',
    'Ngurah Rai-Bali',
    'Soekarno Hatta-Jakarta'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        tahun = int(request.form['tahun'])
        bulan = int(request.form['bulan'])
        jenis = int(request.form['jenis'])
        target = request.form['target']

        features = [tahun, bulan, jenis]
        for t in targets:
            features.append(0 if t != target else 1) 

        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]
        result = int(round(prediction[targets.index(target)])) 
        
        result = f"{result:,}".replace(",", ".")
        bulan_map = {
            1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
            7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"
        }
        nama_bulan = bulan_map[bulan]
        jenis_map = {
            1: "Domestik",
            0: "Internasional"
        }
        jenis = jenis_map[jenis]
        return render_template('index.html', prediction=result, target=target, tahun=tahun, bulan=nama_bulan, jenis=jenis)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
