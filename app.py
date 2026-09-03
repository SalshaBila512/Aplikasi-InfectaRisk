import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="InfectaRisk",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS PREMIUM UI
# =====================================================
st.markdown("""
<style>

.stApp{
    background:#f7f4ef;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#3d342f,#2e2825);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

.block-container{
    padding-top:1rem;
    max-width:1450px;
}

.card{
    background:white;
    border-radius:16px;
    padding:10px;
    border:1px solid #ddd;
    margin-bottom:10px;
}

.title-box{
    background:#dff0d8;
    border:2px solid #7ba67b;
    border-radius:10px;
    text-align:center;
    padding:10px;
    font-size:24px;
    font-weight:bold;
    margin-bottom:15px;
}

.hero{
    background:white;
    border-radius:18px;
    padding:25px;
    border:1px solid #ccc;
    margin-bottom:15px;
}

.stButton > button{
    width:100%;
    border:none;
    border-radius:10px;
    background:#6b584d;
    color:white;
    font-weight:bold;
    padding:12px;
}

.stButton > button:hover{
    background:#4e4038;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.markdown("""
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
""", unsafe_allow_html=True)

# Judul Sidebar
st.sidebar.markdown("""
<h2 style='margin-bottom:5px;'>
<i class="bi bi-heart-pulse-fill"></i> InfectaRisk
</h2>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("""
<p style='font-size:18px; margin-bottom:5px;'>
<i class="bi bi-compass-fill"></i> Navigation
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("Pilih menu sistem")

# Menu Navigasi
menu = st.sidebar.radio(
    "",
    [
        "Home",
        "Proses & Analisis",
        "Input Gejala & Prediksi",
        "Riwayat Pasien"
    ],
    label_visibility="collapsed"
)

# =====================================================
# FUNCTION
# =====================================================
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

# =====================================================
# HOME
# =====================================================
st.markdown("""
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
""", unsafe_allow_html=True)
if menu == "Home":

    st.markdown("<div class='title-box'style='font-size:18px; padding:8px; font-weight:700;'>1. HALAMAN HOME</div>", unsafe_allow_html=True)

    st.markdown("<div class='hero'>", unsafe_allow_html=True)

    col1, col2 = st.columns([4,1])

    with col1:
        st.title("InfectaRisk")
        st.subheader("Sistem Klasifikasi Risiko Penyakit Menular")
        st.write("Analisis cerdas berbasis metode Naïve Bayes")

    with col2:
        st.markdown("""
        <div style='text-align:center; margin-top:10px;'>
            <i class="bi bi-clipboard2-pulse-fill"
            style='font-size:85px; color:#5d524a;'></i>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Informasi Penyakit")
        st.write("""
        Penyakit menular adalah penyakit yang dapat berpindah dari satu orang ke orang lain, baik secara langsung maupun tidak langsung.
        Penularan dapat terjadi melalui udara, kontak langsung, makanan/minuman terkontaminasi,
        serta gigitan hewan tertentu.
        Contoh: TBC, DBD, ISPA, Influenza dan Diare.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Pencegahan")
        st.write("""
        ✔ Rajin mencuci tangan  
        ✔ Gunakan masker saat sakit  
        ✔ Menjaga kebersihan lingkungan  
        ✔ Makan bergizi  
        ✔ Istirahat cukup
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Ilustrasi Penyakit")
    st.write("""
    🫁 **TBC** : Menyerang paru-paru dan menular lewat udara.  
    🦟 **DBD** : Disebabkan gigitan nyamuk Aedes aegypti.  
    😷 **ISPA** : Infeksi saluran pernapasan akut.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Metodologi")
    st.write("""
    🗄️ Data Selection → ⚗️ Preprocessing → 📊 Transformation → 🕸️ Modeling (Naïve Bayes) → ☑️ Evaluation
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PROSES ANALISIS
# =====================================================
elif menu == "Proses & Analisis":

    st.markdown("<div class='title-box'style='font-size:18px; padding:8px; font-weight:700;'>2. PROSES & ANALISIS</div>", unsafe_allow_html=True)

    st.subheader("⚙️ Tahapan Proses Data Mining")

    try:
        df = pd.read_excel("data.xlsx")
        df.columns = df.columns.str.strip()

        # Progress
        st.progress(100)
        st.caption("Seluruh tahapan data mining berhasil dijalankan")

        # 1 Data Selection
        
        st.write("### 1️⃣ Data Selection")
        st.write("""
        Tahap ini mengambil dataset pasien dari file Excel.
        Data berisi gejala G1-G14 dan diagnosis akhir pasien.
        """)
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

        # Grafik distribusi kelas
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 📊 Distribusi Data Diagnosis")

        fig, ax = plt.subplots(figsize=(6,3))
        df["Diagnosis"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah Data")
        plt.tight_layout()
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig, use_container_width=False)

        st.markdown("</div>", unsafe_allow_html=True)

        # 2 Preprocessing
        df = df.drop(columns=[
            "No", "No Rekam Medis", "Tanggal", "Umur"
        ])

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 2️⃣ Preprocessing")
        st.write("""
        Menghapus atribut yang tidak relevan agar model fokus
        pada gejala dan diagnosis.
        """)
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

        # 3 Transformation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 3️⃣ Transformation")
        st.write("""
        Data gejala sudah dalam bentuk numerik:
        
        - 1 = gejala dialami
        - 0 = gejala tidak dialami

        Sehingga langsung siap diproses model.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        # 4 Training
        prior, likelihood, classes = train_naive_bayes(df)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 4️⃣ Training Naïve Bayes")
        st.write("""
        Sistem menghitung probabilitas prior tiap kelas penyakit
        berdasarkan data latih.
        """)
        st.write(prior)
        st.markdown("</div>", unsafe_allow_html=True)

        # 5 Evaluasi
        benar = 0

        for i in range(len(df)):
            row = df.iloc[i]
            input_data = row.drop("Diagnosis").to_dict()

            pred = predict_naive_bayes(input_data, prior, likelihood, classes)
            hasil = max(pred, key=pred.get)

            if hasil == row["Diagnosis"]:
                benar += 1

        akurasi = benar / len(df)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 5️⃣ Evaluasi Model")
        st.write("""
        Pengujian dilakukan dengan membandingkan hasil prediksi
        sistem dengan label asli dataset.
        """)

        st.success(f"Akurasi Model: {akurasi:.2%}")
        st.progress(int(akurasi * 100))

        st.info(f"""
        Berdasarkan hasil pengujian, model Naïve Bayes menghasilkan
        tingkat akurasi sebesar {akurasi:.2%}, sehingga memiliki
        performa sangat baik dalam klasifikasi penyakit menular.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# =====================================================
# INPUT PREDIKSI
# =====================================================
elif menu == "Input Gejala & Prediksi":

    st.markdown(
        "<div class='title-box'style='font-size:18px; padding:8px; font-weight:700;'>3. INPUT GEJALA & PREDIKSI</div>",
        unsafe_allow_html=True
    )

    st.subheader("🧍‍♂️ Input Data Pasien")

    nama = st.text_input("Nama")
    umur = st.number_input("Umur", 0, 120)

    # pilihan simpan data
    simpan_data = st.checkbox("Simpan data pasien ke riwayat")

    st.write("### 📋 Pilih Gejala")

    gejala = {}
    cols = st.columns(2)

    label_gejala = [
        "Batuk > 2 minggu",
        "Batuk Berdarah",
        "Demam Lama",
        "Keringat Malam",
        "BB Turun",
        "Demam Tinggi",
        "Nyeri Sendi",
        "Mual/Muntah",
        "Ruam Kulit",
        "Sakit Kepala",
        "Batuk Kering/Berdahak",
        "Pilek",
        "Sakit Tenggorokan",
        "Sesak Nafas"
    ]

    # checkbox gejala
    gejala = {}

    col1, col2 = st.columns(2)

    for i, label in enumerate(label_gejala):
        kode = f"G{i+1}"

        if i < 7:
            with col1:
                gejala[kode] = st.checkbox(label, key=kode)
        else:
            with col2:
                gejala[kode] = st.checkbox(label, key=kode)

    # tombol analisis
    if st.button("🔍 Analisis Sekarang"):

        # validasi nama
        if nama.strip() == "":
            st.warning("Nama pasien harus diisi")
            st.stop()

        try:

            # baca dataset training
            df = pd.read_excel("data.xlsx")
            df.columns = df.columns.str.strip()

            # hapus kolom tidak relevan
            df = df.drop(columns=[
                "No",
                "No Rekam Medis",
                "Tanggal",
                "Umur"
            ])

            # training model
            prior, likelihood, classes = train_naive_bayes(df)

            # input gejala
            input_data = {
                k: int(v)
                for k, v in gejala.items()
            }

            # prediksi
            posterior = predict_naive_bayes(
                input_data,
                prior,
                likelihood,
                classes
            )

            hasil = max(posterior, key=posterior.get)

            # probabilitas %
            exp_values = {
                k: np.exp(v)
                for k, v in posterior.items()
            }

            total = sum(exp_values.values())

            persen = {
                k: (v / total) * 100
                for k, v in exp_values.items()
            }

            # warna hasil
            warna = "#16A34A" if hasil == "NEGATIF" else "#DC2626" 

            st.markdown(f""" 
            <div class='card'> 
            <h3>📈 Hasil Diagnosis</h3> <p style='font-size:38px; font-weight:700; color:{warna};
            margin:0;'>{hasil}</p> 
            </div>
            """, unsafe_allow_html=True)


            # probabilitas
            st.write("### 📊 Probabilitas")

            for k, v in persen.items():
                st.progress(int(v))
                st.write(f"{k}: {v:.2f}%")

            # =====================================================
            # SIMPAN DATA PASIEN
            # =====================================================

            if simpan_data:

                data_pasien = {
                    "Nama": nama,
                    "Umur": umur
                }

                # simpan gejala
                for k, v in gejala.items():
                    data_pasien[k] = int(v)

                # simpan diagnosis
                data_pasien["Diagnosis"] = hasil

                # dataframe baru
                df_baru = pd.DataFrame([data_pasien])

                file_riwayat = "hasil_pasien.xlsx"

                # jika file sudah ada
                if os.path.exists(file_riwayat):

                    df_lama = pd.read_excel(file_riwayat)

                    df_gabung = pd.concat(
                        [df_lama, df_baru],
                        ignore_index=True
                    )

                    df_gabung.to_excel(
                        file_riwayat,
                        index=False
                    )

                # jika file belum ada
                else:

                    df_baru.to_excel(
                        file_riwayat,
                        index=False
                    )

                st.success("Data pasien berhasil disimpan")

            # =====================================================
            # PENJELASAN HASIL
            # =====================================================

            st.markdown("### 🧠 Penjelasan Hasil")

            if hasil == "NEGATIF":

                st.write(f"""
                Berdasarkan gejala yang dipilih,
                sistem menghitung probabilitas
                menggunakan metode Naïve Bayes.

                Hasil menunjukkan bahwa pasien
                memiliki kemungkinan tertinggi
                dalam kategori **NEGATIF**,
                sehingga tidak terindikasi
                TBC, DBD, maupun ISPA.
                """)

            else:

                st.write(f"""
                Berdasarkan gejala yang dipilih,
                sistem menghitung probabilitas
                menggunakan metode Naïve Bayes.

                Hasil menunjukkan bahwa pasien
                memiliki kemungkinan tertinggi
                mengalami **{hasil}**.
                """)

        except Exception as e:
            st.error(f"Error: {e}")

# =====================================================
# RIWAYAT
# =====================================================
elif menu == "Riwayat Pasien":

    st.markdown("<div class='title-box'style='font-size:18px; padding:8px; font-weight:700;'>4. RIWAYAT DATA PASIEN</div>", unsafe_allow_html=True)

    file = "hasil_pasien.xlsx"

    if os.path.exists(file):

        df = pd.read_excel(file)
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ Hapus Semua Riwayat"):
            os.remove(file)
            st.success("Riwayat berhasil dihapus")
            st.rerun()

    else:
        st.warning("Belum ada data")

# =====================================================
# FOOTER
# =====================================================
st.markdown(
"""
<hr>
<center>
Created by Salsha Billa Tiara Anggraeni | 2026
</center>
""",
unsafe_allow_html=True
)