
import os
import joblib
import pandas as pd
import sklearn

# ==================================================
# 1. Memeriksa file yang diperlukan
# ==================================================

FILE_MODEL = "epl_logistic_regression.pkl"
FILE_FITUR = "epl_feature_columns.pkl"
FILE_DATA = "epl_matches_clean.csv"

file_wajib = [FILE_MODEL, FILE_FITUR, FILE_DATA]

for nama_file in file_wajib:
    if not os.path.exists(nama_file):
        raise FileNotFoundError(
            f"File '{nama_file}' tidak ditemukan. "
            "Pastikan file berada di folder proyek."
        )

print("Semua file utama ditemukan.")

# ==================================================
# 2. Memuat model dan urutan fitur
# ==================================================

model_prediksi = joblib.load(FILE_MODEL)
kolom_fitur = joblib.load(FILE_FITUR)

print("Model berhasil dimuat.")
print("Library scikit-learn:", sklearn.__version__)
print("Urutan fitur model:", kolom_fitur)

# ==================================================
# 3. Memuat data historis EPL
# ==================================================

data_epl = pd.read_csv(FILE_DATA)

print("\nData historis berhasil dimuat.")
print("Jumlah baris:", len(data_epl))
print("Nama kolom:", data_epl.columns.tolist())

# ==================================================
# 4. Menyiapkan satu contoh prediksi
#    Angka berikut hanya contoh ilustratif.
# ==================================================

contoh_pertandingan = pd.DataFrame(
    [
        {
            "HomeGoalsAvg5": 1.8,
            "HomeConcededAvg5": 0.8,
            "HomePointsAvg5": 2.2,
            "AwayGoalsAvg5": 1.2,
            "AwayConcededAvg5": 1.4,
            "AwayPointsAvg5": 1.0,
        }
    ],
    columns=kolom_fitur,
)

print("\nFitur yang akan diuji:")
print(contoh_pertandingan.to_string(index=False))

# ==================================================
# 5. Melakukan prediksi
# ==================================================

kode_prediksi = model_prediksi.predict(contoh_pertandingan)[0]
probabilitas = model_prediksi.predict_proba(
    contoh_pertandingan
)[0]

terjemahan_hasil = {
    "H": "Menang Kandang",
    "D": "Seri",
    "A": "Menang Tandang",
}

print("\n========== HASIL PENGUJIAN ==========")
print("Kode hasil:", kode_prediksi)
print("Prediksi:", terjemahan_hasil.get(
    kode_prediksi, "Kode hasil tidak dikenal"
))

print("\nProbabilitas prediksi:")

for kelas, nilai in zip(model_prediksi.classes_, probabilitas):
    nama_hasil = terjemahan_hasil.get(kelas, kelas)
    print(f"{nama_hasil}: {nilai * 100:.2f}%")

print("\nTotal probabilitas:", f"{probabilitas.sum() * 100:.2f}%")

print("\nPENGUJIAN SELESAI.")
