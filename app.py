
# ============================================================
# WEBSITE PREDIKSI HASIL PERTANDINGAN EPL
# Model: Logistic Regression (class_weight="balanced")
# Framework: Streamlit
# ============================================================

import os
from datetime import date

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. KONFIGURASI HALAMAN WEBSITE
# ============================================================

st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide",
)

# Nama file yang harus berada di folder yang sama dengan app.py
FILE_MODEL = "epl_logistic_regression.pkl"
FILE_FITUR = "epl_feature_columns.pkl"
FILE_DATA = "epl_matches_clean.csv"

# Urutan fitur harus konsisten dengan model yang sudah dilatih
KOLOM_FITUR = [
    "HomeGoalsAvg5",
    "HomeConcededAvg5",
    "HomePointsAvg5",
    "AwayGoalsAvg5",
    "AwayConcededAvg5",
    "AwayPointsAvg5",
]

TERJEMAHAN_HASIL = {
    "H": "Menang Kandang",
    "D": "Seri",
    "A": "Menang Tandang",
}


# ============================================================
# 2. MEMUAT MODEL DAN DATA HISTORIS
# ============================================================

@st.cache_resource
def muat_model():
    """Memuat model dan daftar fitur satu kali."""

    if not os.path.exists(FILE_MODEL):
        raise FileNotFoundError(
            f"File model '{FILE_MODEL}' tidak ditemukan."
        )

    if not os.path.exists(FILE_FITUR):
        raise FileNotFoundError(
            f"File fitur '{FILE_FITUR}' tidak ditemukan."
        )

    model = joblib.load(FILE_MODEL)
    kolom_fitur = joblib.load(FILE_FITUR)

    if list(kolom_fitur) != KOLOM_FITUR:
        raise ValueError(
            "Urutan atau nama fitur dalam file model "
            "tidak sesuai dengan aplikasi."
        )

    kelas_model = set(model.classes_)

    if kelas_model != {"H", "D", "A"}:
        raise ValueError(
            "Model harus memiliki kelas hasil H, D, dan A. "
            f"Kelas yang ditemukan: {kelas_model}"
        )

    return model, kolom_fitur


@st.cache_data
def muat_data_historis():
    """Membaca dan menyiapkan data pertandingan EPL."""

    if not os.path.exists(FILE_DATA):
        raise FileNotFoundError(
            f"File data '{FILE_DATA}' tidak ditemukan."
        )

    data = pd.read_csv(FILE_DATA)

    kolom_wajib = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]

    kolom_hilang = [
        kolom
        for kolom in kolom_wajib
        if kolom not in data.columns
    ]

    if kolom_hilang:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {kolom_hilang}"
        )

    # Data Football-Data dapat memakai format tanggal DD/MM/YYYY.
    # dayfirst=True membantu membaca format tersebut.
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
        dayfirst=True,
    )

    data["FTHG"] = pd.to_numeric(
        data["FTHG"], errors="coerce"
    )

    data["FTAG"] = pd.to_numeric(
        data["FTAG"], errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]
    ).copy()

    data["FTR"] = data["FTR"].astype(str).str.strip()

    # Hanya gunakan hasil pertandingan yang valid.
    data = data[data["FTR"].isin(["H", "D", "A"])].copy()

    if data.empty:
        raise ValueError(
            "Tidak ada data pertandingan valid yang dapat digunakan."
        )

    return data.sort_values("Date").reset_index(drop=True)


# Tampilkan pesan yang jelas jika file atau model bermasalah.
try:
    model_prediksi, kolom_fitur = muat_model()
    data_historis = muat_data_historis()

except Exception as error:
    st.error(
        "Aplikasi tidak dapat memuat model atau data historis."
    )
    st.exception(error)
    st.stop()


# ============================================================
# 3. MENGHITUNG STATISTIK LIMA PERTANDINGAN TERAKHIR
# ============================================================

def hitung_statistik_lima_pertandingan(
    nama_tim,
    tanggal_prediksi,
    data,
):
    """
    Menghitung rata-rata gol dicetak, gol kebobolan,
    dan poin dari lima pertandingan terakhir tim.

    Hanya pertandingan dengan tanggal sebelum tanggal prediksi
    yang digunakan untuk mencegah kebocoran informasi.
    """

    tanggal_prediksi = pd.to_datetime(tanggal_prediksi)

    # Hanya gunakan pertandingan sebelum tanggal prediksi.
    data_sebelumnya = data[
        data["Date"] < tanggal_prediksi
    ].copy()

    # Pertandingan saat tim bermain sebagai tuan rumah.
    kandang = data_sebelumnya[
        data_sebelumnya["HomeTeam"] == nama_tim
    ].copy()

    kandang["Gol_Dicetak"] = kandang["FTHG"]
    kandang["Gol_Kebobolan"] = kandang["FTAG"]

    kandang["Poin"] = kandang["FTR"].map({
        "H": 3,
        "D": 1,
        "A": 0,
    })

    # Pertandingan saat tim bermain sebagai tim tandang.
    tandang = data_sebelumnya[
        data_sebelumnya["AwayTeam"] == nama_tim
    ].copy()

    tandang["Gol_Dicetak"] = tandang["FTAG"]
    tandang["Gol_Kebobolan"] = tandang["FTHG"]

    tandang["Poin"] = tandang["FTR"].map({
        "A": 3,
        "D": 1,
        "H": 0,
    })

    # Gabungkan seluruh pertandingan tim.
    riwayat_tim = pd.concat(
        [kandang, tandang],
        ignore_index=True,
    )

    riwayat_tim = riwayat_tim.sort_values(
        "Date",
        ascending=False,
    )

    lima_pertandingan = riwayat_tim.head(5)

    if len(lima_pertandingan) < 5:
        raise ValueError(
            f"Tim {nama_tim} hanya memiliki "
            f"{len(lima_pertandingan)} pertandingan "
            "sebelum tanggal prediksi. "
            "Diperlukan minimal lima pertandingan."
        )

    kolom_statistik = [
        "Gol_Dicetak",
        "Gol_Kebobolan",
        "Poin",
    ]

    if lima_pertandingan[kolom_statistik].isnull().any().any():
        raise ValueError(
            f"Statistik lima pertandingan tim {nama_tim} "
            "mengandung data kosong."
        )

    return {
        "GoalsAvg5": lima_pertandingan[
            "Gol_Dicetak"
        ].mean(),

        "ConcededAvg5": lima_pertandingan[
            "Gol_Kebobolan"
        ].mean(),

        "PointsAvg5": lima_pertandingan[
            "Poin"
        ].mean(),
    }


# ============================================================
# 4. MEMBUAT FITUR PERTANDINGAN
# ============================================================

def buat_fitur_pertandingan(
    tim_kandang,
    tim_tandang,
    tanggal_prediksi,
    data,
):
    """Mengubah riwayat kedua tim menjadi enam fitur model."""

    if tim_kandang == tim_tandang:
        raise ValueError(
            "Tim kandang dan tim tandang harus berbeda."
        )

    statistik_kandang = hitung_statistik_lima_pertandingan(
        nama_tim=tim_kandang,
        tanggal_prediksi=tanggal_prediksi,
        data=data,
    )

    statistik_tandang = hitung_statistik_lima_pertandingan(
        nama_tim=tim_tandang,
        tanggal_prediksi=tanggal_prediksi,
        data=data,
    )

    fitur = {
        "HomeGoalsAvg5": statistik_kandang["GoalsAvg5"],
        "HomeConcededAvg5": statistik_kandang["ConcededAvg5"],
        "HomePointsAvg5": statistik_kandang["PointsAvg5"],

        "AwayGoalsAvg5": statistik_tandang["GoalsAvg5"],
        "AwayConcededAvg5": statistik_tandang["ConcededAvg5"],
        "AwayPointsAvg5": statistik_tandang["PointsAvg5"],
    }

    # Pastikan urutan kolom sama dengan urutan saat pelatihan.
    return pd.DataFrame(
        [fitur],
        columns=kolom_fitur,
    )


# ============================================================
# 5. FUNGSI PREDIKSI
# ============================================================

def prediksi_pertandingan(
    tim_kandang,
    tim_tandang,
    tanggal_prediksi,
    data,
):
    """Menghasilkan prediksi dan probabilitas tiga kelas."""

    fitur = buat_fitur_pertandingan(
        tim_kandang=tim_kandang,
        tim_tandang=tim_tandang,
        tanggal_prediksi=tanggal_prediksi,
        data=data,
    )

    kode_hasil = model_prediksi.predict(fitur)[0]

    nilai_probabilitas = model_prediksi.predict_proba(
        fitur
    )[0]

    probabilitas = {
        TERJEMAHAN_HASIL[kelas]: float(nilai)
        for kelas, nilai in zip(
            model_prediksi.classes_,
            nilai_probabilitas,
        )
    }

    return kode_hasil, probabilitas, fitur


# ============================================================
# 6. TAMPILAN UTAMA WEBSITE
# ============================================================

st.title("⚽ EPL Match Predictor")

st.write(
    "Website prediksi hasil pertandingan English Premier League "
    "menggunakan Machine Learning Logistic Regression."
)

st.caption(
    "Model menggunakan statistik lima pertandingan terakhir "
    "setiap tim dan memprediksi menang kandang, seri, atau "
    "menang tandang."
)

# Informasi sumber data.
tanggal_terakhir = data_historis["Date"].max().date()

st.info(
    f"Data historis tersedia sampai {tanggal_terakhir:%d %B %Y}. "
    "Prediksi menggunakan data yang tersedia sebelum tanggal "
    "pertandingan yang dipilih. Data tidak diperbarui otomatis."
)

st.divider()

# Mengumpulkan semua nama tim dari data historis.
daftar_tim = sorted(
    set(data_historis["HomeTeam"].dropna())
    | set(data_historis["AwayTeam"].dropna())
)

if len(daftar_tim) < 2:
    st.error(
        "Data historis harus memuat setidaknya dua tim."
    )
    st.stop()


# ============================================================
# 7. INPUT PENGGUNA
# ============================================================

st.subheader("Pilih Pertandingan")

kolom_kandang, kolom_tandang = st.columns(2)

with kolom_kandang:
    tim_kandang = st.selectbox(
        "Tim Kandang",
        options=daftar_tim,
        index=0,
        help="Tim yang bermain di kandang.",
    )

with kolom_tandang:
    indeks_tandang = 1 if len(daftar_tim) > 1 else 0

    tim_tandang = st.selectbox(
        "Tim Tandang",
        options=daftar_tim,
        index=indeks_tandang,
        help="Tim yang bermain sebagai tamu.",
    )

tanggal_prediksi = st.date_input(
    "Tanggal Prediksi",
    value=date.today(),
    help=(
        "Statistik yang digunakan hanya berasal dari pertandingan "
        "sebelum tanggal ini."
    ),
)

st.caption(
    "Pilih tanggal pertandingan yang ingin dianalisis. "
    "Tanggal ini tidak otomatis memverifikasi jadwal resmi EPL."
)

tombol_prediksi = st.button(
    "Prediksi Pertandingan",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 8. MENAMPILKAN HASIL PREDIKSI
# ============================================================

if tombol_prediksi:

    if tim_kandang == tim_tandang:
        st.error(
            "Pilih dua tim yang berbeda untuk melakukan prediksi."
        )

    else:
        try:
            kode_hasil, probabilitas, fitur = (
                prediksi_pertandingan(
                    tim_kandang=tim_kandang,
                    tim_tandang=tim_tandang,
                    tanggal_prediksi=tanggal_prediksi,
                    data=data_historis,
                )
            )

            nama_hasil = TERJEMAHAN_HASIL[kode_hasil]

            st.divider()
            st.subheader("Hasil Prediksi")

            st.write(
                f"**{tim_kandang}** vs **{tim_tandang}**"
            )

            st.success(
                f"Prediksi model: {nama_hasil}"
            )

            # Tiga metrik probabilitas.
            kolom_home, kolom_draw, kolom_away = st.columns(3)

            with kolom_home:
                st.metric(
                    label="Menang Kandang",
                    value=f"{probabilitas['Menang Kandang']:.2%}",
                )

            with kolom_draw:
                st.metric(
                    label="Seri",
                    value=f"{probabilitas['Seri']:.2%}",
                )

            with kolom_away:
                st.metric(
                    label="Menang Tandang",
                    value=f"{probabilitas['Menang Tandang']:.2%}",
                )

            # Menyiapkan grafik probabilitas.
            data_grafik = pd.DataFrame({
                "Hasil": [
                    "Menang Kandang",
                    "Seri",
                    "Menang Tandang",
                ],
                "Probabilitas (%)": [
                    probabilitas["Menang Kandang"] * 100,
                    probabilitas["Seri"] * 100,
                    probabilitas["Menang Tandang"] * 100,
                ],
            }).set_index("Hasil")

            st.subheader("Visualisasi Probabilitas")

            st.bar_chart(data_grafik)

            st.caption(
                "Probabilitas adalah estimasi model, bukan jaminan "
                "hasil pertandingan. Ketepatan probabilitas juga "
                "bergantung pada kualitas data dan kalibrasi model."
            )

            # Statistik yang menjadi input model.
            with st.expander(
                "Lihat Statistik Lima Pertandingan Terakhir"
            ):
                tabel_fitur = fitur.T.reset_index()
                tabel_fitur.columns = [
                    "Nama Fitur",
                    "Nilai",
                ]

                tabel_fitur["Nilai"] = (
                    tabel_fitur["Nilai"].round(2)
                )

                st.dataframe(
                    tabel_fitur,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Nilai merupakan rata-rata lima pertandingan "
                    "sebelum tanggal prediksi, bukan prediksi skor "
                    "akhir pertandingan."
                )

        except ValueError as error:
            st.warning(str(error))

        except Exception as error:
            st.error(
                "Terjadi kesalahan ketika memproses prediksi."
            )
            st.exception(error)


# ============================================================
# 9. INFORMASI TAMBAHAN
# ============================================================

with st.expander("Tentang Model dan Data"):

    st.write(
        "**Algoritma:** Logistic Regression dengan "
        "`class_weight='balanced'`."
    )

    st.write(
        "**Fitur model:** rata-rata gol dicetak, gol kebobolan, "
        "dan poin dari lima pertandingan terakhir untuk "
        "masing-masing tim."
    )

    st.write(
        "**Kelas hasil:** H = menang kandang, D = seri, "
        "A = menang tandang."
    )

    st.write(
        "**Batasan:** aplikasi menggunakan data historis yang "
        "tersedia secara lokal. Aplikasi tidak memeriksa jadwal "
        "resmi, cedera pemain, susunan pemain, atau pembaruan "
        "hasil pertandingan secara langsung."
    )
