import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Klasifikasi Penyakit", layout="wide")

st.title("🩺 Aplikasi Klasifikasi Risiko Penyakit Menular")
st.caption("Penerapan Algoritma Naïve Bayes (Dynamic Training)")

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Proses & Analisis", "Input Gejala & Prediksi"]
)

# =========================
# FUNGSI NAIVE BAYES
# =========================
def train_naive_bayes(df):
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    classes = y.unique()
    prior = {}
    likelihood = {}

    for c in classes:
        df_c = df[y == c]
        prior[c] = len(df_c) / len(df)

        likelihood[c] = {}
        for col in X.columns:
            prob_1 = (df_c[col].sum() + 1) / (len(df_c) + 2)
            likelihood[c][col] = prob_1

    return prior, likelihood, classes


def predict_naive_bayes(input_data, prior, likelihood, classes):
    posterior = {}

    for c in classes:
        prob = np.log(prior[c])

        for feature, value in input_data.items():
            if value == 1:
                prob += np.log(likelihood[c][feature])
            else:
                prob += np.log(1 - likelihood[c][feature])

        posterior[c] = prob

    return posterior


# =========================
# HOME
# =========================
if menu == "Home":

    st.subheader("👋 Selamat Datang")

    st.markdown("""
    Aplikasi ini digunakan untuk membantu klasifikasi risiko penyakit menular 
    menggunakan metode **Naïve Bayes**.

    Sistem bekerja berdasarkan input gejala pasien dan data historis yang tersimpan dalam file Excel.
    """)

    st.markdown("### 🧠 Metodologi")
    st.markdown("""
    1. Data Selection  
    2. Preprocessing  
    3. Transformation  
    4. Modeling (Naïve Bayes)  
    5. Evaluation  
    """)

# =========================
# PROSES & ANALISIS
# =========================
elif menu == "Proses & Analisis":

    st.header("⚙️ Tahapan Proses Data Mining")

    try:
        # 1. DATA SELECTION
        st.subheader("1️⃣ Data Selection")
        df = pd.read_excel("data.xlsx")
        st.dataframe(df.head())

        # 2. PREPROCESSING
        st.subheader("2️⃣ Preprocessing")
        df = df.drop(columns=[
            "No", "No Rekam Medis", "Tanggal", "Nama Lengkap"
        ])
        st.write("Menghapus atribut tidak relevan")
        st.dataframe(df.head())

        # 3. TRANSFORMATION
        st.subheader("3️⃣ Transformation")
        st.write("Data gejala sudah dalam bentuk numerik (0 dan 1), sehingga tidak memerlukan encoding tambahan.")

        # 4. TRAINING
        st.subheader("4️⃣ Training Naïve Bayes")
        prior, likelihood, classes = train_naive_bayes(df)

        st.write("Probabilitas Prior:")
        st.write(prior)

        # 5. EVALUASI
        st.subheader("5️⃣ Evaluasi Model")

        benar = 0

        for i in range(len(df)):
            row = df.iloc[i]
            input_data = row.drop("Diagnosis").to_dict()

            pred = predict_naive_bayes(input_data, prior, likelihood, classes)
            hasil = max(pred, key=pred.get)

            if hasil == row["Diagnosis"]:
                benar += 1

        akurasi = benar / len(df)

        st.success(f"Akurasi Model: {akurasi:.2%}")

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# INPUT & PREDIKSI
# =========================
elif menu == "Input Gejala & Prediksi":

    st.header("🧍‍♂️ Input Data Pasien")

    nama = st.text_input("Nama")
    umur = st.number_input("Umur (opsional)", min_value=0, max_value=120)

    st.subheader("📋 Pilih Gejala")

    gejala = {}
    cols = st.columns(2)

    label_gejala = {
        "G1": "Batuk > 2 minggu",
        "G2": "Batuk Berdarah",
        "G3": "Demam Lama",
        "G4": "Keringat Malam",
        "G5": "BB Turun",
        "G6": "Demam Tinggi",
        "G7": "Nyeri Sendi",
        "G8": "Mual/Muntah",
        "G9": "Ruam Kulit",
        "G10": "Trombosit Rendah",
        "G11": "Batuk",
        "G12": "Pilek",
        "G13": "Sakit Tenggorokan",
        "G14": "Sesak Nafas"
    }

    for i, (kode, label) in enumerate(label_gejala.items()):
        with cols[i % 2]:
            gejala[kode] = st.checkbox(f"{kode} - {label}")

    if st.button("🔍 Analisis"):

        try:
            df = pd.read_excel("data.xlsx")

            df = df.drop(columns=[
                "No", "No Rekam Medis", "Tanggal", "Nama Lengkap"
            ])

            prior, likelihood, classes = train_naive_bayes(df)

            input_data = {k: int(v) for k, v in gejala.items()}
            posterior = predict_naive_bayes(input_data, prior, likelihood, classes)

            hasil = max(posterior, key=posterior.get)

            # KONVERSI KE PERSEN
            exp_values = {k: np.exp(v) for k, v in posterior.items()}
            total = sum(exp_values.values())
            persen = {k: (v / total) * 100 for k, v in exp_values.items()}

            st.subheader("📈 Hasil Klasifikasi")
            st.success(f"Hasil Diagnosis: {hasil}")

            st.subheader("📊 Probabilitas (%)")
            for k, v in persen.items():
                st.write(f"{k}: {v:.2f}%")

            st.markdown("### 🧠 Penjelasan Hasil")
            st.write(f"""
            Berdasarkan gejala yang dipilih, sistem menghitung probabilitas setiap kategori penyakit 
            menggunakan metode Naïve Bayes.

            Hasil menunjukkan bahwa pasien memiliki kemungkinan tertinggi mengalami **{hasil}**.
            """)

        except Exception as e:
            st.error(f"Error: {e}")