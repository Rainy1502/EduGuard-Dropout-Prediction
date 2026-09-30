<div align="center">
  <img src="assets/favicon.svg" alt="EduGuard Logo" width="90"/>

  # EduGuard: Deteksi Dini Mahasiswa Berisiko Dropout

  **Proyek Akhir Belajar Penerapan Data Science · Jaya Jaya Institut**

  [![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
  [![pandas](https://img.shields.io/badge/pandas-3.0-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/python/)
  [![Metabase](https://img.shields.io/badge/Metabase-0.63-509EE3.svg?logo=metabase&logoColor=white)](https://www.metabase.com/)
  [![Docker](https://img.shields.io/badge/Docker-required_for_Metabase-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

  [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[nama-app].streamlit.app)
</div>

> Sistem peringatan dini untuk menemukan mahasiswa yang berisiko **dropout** sebelum terlambat.
>
> Proyek ini mencakup analisis data 4.424 mahasiswa Jaya Jaya Institut, model machine learning yang memprediksi risiko dropout (recall 81%, ROC-AUC 0,93), prototype aplikasi Streamlit **EduGuard**, serta business dashboard **Metabase** untuk memonitor faktor-faktor penyebab dropout.

---

## 📌 Tentang Proyek

| Detail | Keterangan |
|---|---|
| **Nama** | [Nama Lengkap] |
| **Email** | [Email] |
| **ID Dicoding** | [username_dicoding] |
| **Institusi (fiktif)** | Jaya Jaya Institut |
| **Dataset** | Students' Performance: 4.424 mahasiswa, 36 fitur, target `Status` (Dropout / Enrolled / Graduate) |
| **Target model** | Dropout vs Non-Dropout (Graduate + Enrolled) |
| **Model** | Logistic Regression, 20 fitur + 2 fitur turunan, threshold 0,51 |
| **Performa (data uji)** | Recall 81,3% · Precision 80,5% · F1 0,809 · Akurasi 87,7% · ROC-AUC 0,929 |
| **Prototype** | Streamlit (EduGuard), di-*deploy* ke Streamlit Community Cloud |
| **Dashboard** | Metabase (SQLite sebagai sumber data) |

---

## 💼 Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berdiri sejak tahun 2000 dan telah menghasilkan banyak lulusan dengan reputasi baik. Namun, jumlah mahasiswa yang tidak menyelesaikan pendidikannya (**dropout**) masih tinggi. Kondisi ini merugikan institusi dari sisi reputasi, akreditasi, dan pendapatan, sekaligus merugikan mahasiswa itu sendiri.

Jaya Jaya Institut ingin **mendeteksi sedini mungkin mahasiswa yang berpotensi dropout** agar dapat diberi bimbingan khusus, serta membutuhkan dashboard untuk memahami data dan memonitor performa mahasiswa.

### Permasalahan Bisnis

1. Seberapa besar tingkat dropout di Jaya Jaya Institut?
2. Faktor apa saja (akademik, finansial, demografis, program studi, dan jalur masuk) yang paling berkaitan dengan dropout?
3. Bagaimana mendeteksi mahasiswa yang berisiko dropout sejak dini secara otomatis?
4. Bagaimana institusi dapat memonitor performa mahasiswa secara berkelanjutan?

### Cakupan Proyek

| Tahap | Output |
|---|---|
| **Data understanding & EDA** | Faktor utama penyebab dropout (`notebook.ipynb`) |
| **Data preparation** | Seleksi 20 fitur, fitur turunan rasio kelulusan MK, pipeline preprocessing |
| **Modeling & evaluation** | Perbandingan 4 algoritma, tuning, pemilihan threshold, evaluasi pada data uji |
| **Deployment** | Prototype Streamlit **EduGuard** di Streamlit Community Cloud |
| **Business dashboard** | Dashboard Metabase 4 tab dengan filter program studi & gender |
| **Rekomendasi** | Kesimpulan dan action items bagi institusi |

### Persiapan

**Sumber data:** [Students' Performance (Dicoding Academy)](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance), berasal dari dataset UCI *Predict Students' Dropout and Academic Success* (Realinho dkk., 2021). Salinan data ada di `data/data.csv` (pemisah `;`).

**Setup environment** (Python 3.12):

**1. Buat dan aktifkan virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. (Opsional) Jalankan ulang notebook**
```bash
jupyter notebook notebook.ipynb
```
> 💡 Notebook menghasilkan ulang model (`model/`), data dashboard (`data/students_clean.csv`, `data/students.db`), dan contoh input batch (`data/contoh_input_batch.csv`). Semua file tersebut sudah tersedia, jadi langkah ini opsional.

---

## 🗂️ Struktur Proyek

```text
submission/
│
├── 📄 README.md                        # Dokumentasi proyek (file ini)
├── 📓 notebook.ipynb                   # Proses data science lengkap: EDA → modeling → evaluasi
├── 🐍 app.py                           # Prototype Streamlit (EduGuard)
├── 🐍 utils.py                         # Mapping label & feature engineering (dipakai notebook dan app)
├── 📄 requirements.txt                 # Dependencies
├── 🗄️ metabase.db.mv.db                # Database Metabase berisi business dashboard
├── 🖼️ [username_dicoding]-dashboard.png # Screenshot business dashboard
│
├── 📁 data/
│   ├── data.csv                        # Dataset asli
│   ├── students_clean.csv              # Data berlabel (hasil notebook)
│   ├── students.db                     # Database SQLite, sumber data Metabase
│   └── contoh_input_batch.csv          # Template / contoh input prediksi batch
│
├── 📁 model/
│   ├── dropout_model.joblib            # Pipeline: feature engineering + preprocessing + model
│   └── model_metadata.json             # Threshold, metrik, dan feature importance
│
├── 📁 assets/                          # CSS & logo aplikasi
└── 📁 .streamlit/config.toml           # Tema aplikasi
```

---

## 🔬 Alur Kerja Sistem

```
                     [ data/data.csv ]
               4.424 mahasiswa · 36 fitur
                           │
                           ▼
                  [ notebook.ipynb ]
         EDA → preparation → modeling → evaluasi
                           │
          ┌────────────────┴─────────────────┐
          ▼                                  ▼
[ model/dropout_model.joblib ]     [ data/students.db ]
  Logistic Regression pipeline      data berlabel (SQLite)
          │                                  │
          ▼                                  ▼
  [ EduGuard · Streamlit ]           [ Metabase Dashboard ]
  Dashboard · Prediksi individu      4 tab · filter prodi & gender
  Prediksi batch · Tentang model     ← Port 3000
  ← Port 8501 / Streamlit Cloud
```

---

## 📊 Business Dashboard

Dashboard dibuat dengan **Metabase** dan terdiri dari 4 tab. Filter **Program Studi** dan **Gender** berlaku untuk semua grafik.

| Tab | Isi |
|---|---|
| **Ringkasan** | KPI (total mahasiswa, jumlah & persentase dropout, enrolled, graduate, menunggak UKT), distribusi status, dropout rate menurut kondisi mahasiswa, temuan utama |
| **Akademik** | Dropout rate berdasarkan jumlah MK lulus di semester 2, rata-rata MK lulus dan nilai per status |
| **Finansial & Demografi** | Dropout rate berdasarkan kondisi finansial, komposisi status per pembayaran UKT, dropout rate per usia, gender, waktu kuliah, dan status pernikahan |
| **Prodi & Jalur Masuk** | Dropout rate per program studi dan jalur masuk, tabel ringkasan per program studi |

Pada grafik dropout rate, batang **oranye** menandai kelompok dengan dropout rate **di atas rata-rata** segmen yang sedang difilter, sedangkan **abu-abu** di bawah rata-rata.

![Business Dashboard]([username_dicoding]-dashboard.png)

**Akses Metabase**

| | |
|---|---|
| **Email** | `root@mail.com` |
| **Password** | `root123` |

**Cara menjalankan dashboard** (membutuhkan Docker)

```bash
# Jalankan dari folder proyek ini
mkdir metabase-data
cp metabase.db.mv.db metabase-data/     # Windows PowerShell: Copy-Item metabase.db.mv.db metabase-data\

docker run -d -p 3000:3000 \
  -v "$(pwd)/metabase-data:/metabase.db" \
  -v "$(pwd)/data:/data" \
  --name metabase metabase/metabase:v0.63.18.5
```
> 💡 Buka **http://localhost:3000**, login dengan akun di atas, lalu buka dashboard **"Jaya Jaya Institut - Student Performance & Dropout Monitoring"** di menu *Our analytics*. Di Windows PowerShell, ganti `$(pwd)` dengan `${PWD}` dan tulis perintah `docker run` dalam satu baris.

---

## 🤖 Menjalankan Sistem Machine Learning

Prototype sistem machine learning bernama **EduGuard**, dibuat dengan Streamlit.

**🔗 Link prototype:** [https://[nama-app].streamlit.app](https://[nama-app].streamlit.app)

**Menjalankan secara lokal** (setelah langkah *Persiapan*):
```bash
streamlit run app.py
```
> 💡 Aplikasi terbuka di **http://localhost:8501**. Gunakan tombol *Isi contoh risiko rendah/tinggi* di tab Prediksi Individu untuk mencoba dengan cepat.

### Fitur Utama

| Fitur | Detail |
|---|---|
| **Dashboard** | Monitoring dropout per segmen dengan filter program studi, gender, dan waktu kuliah |
| **Prediksi Individu** | Probabilitas dan tingkat risiko dropout (**Rendah** < 25%, **Sedang** 25–51%, **Tinggi** ≥ 51%), faktor yang paling memengaruhi prediksi, dan rekomendasi untuk dosen wali |
| **Prediksi Batch** | Unggah CSV banyak mahasiswa (template di `data/contoh_input_batch.csv`), filter per tingkat risiko, unduh hasil |
| **Tentang Model** | Performa model, alur prediksi, dan pengaruh setiap fitur |

### Model

| Aspek | Keterangan |
|---|---|
| **Algoritma** | Logistic Regression (C = 10), terpilih dari perbandingan dengan Decision Tree, Random Forest, dan Gradient Boosting |
| **Target** | Dropout (1) vs Non-Dropout (0). *Enrolled* (terlambat lulus) digabung ke Non-Dropout karena belum dropout, sehingga seluruh 4.424 data tetap digunakan |
| **Fitur** | 20 fitur (akademik semester 1–2, finansial, profil, pendaftaran) + 2 fitur turunan (rasio kelulusan MK per semester) |
| **Threshold** | 0,51, dipilih agar recall minimal 80% |
| **Performa (data uji)** | Recall 81,3% · Precision 80,5% · F1 0,809 · Akurasi 87,7% · ROC-AUC 0,929 |
| **Validasi tingkat risiko** | Dropout rate aktual pada data uji: Rendah 4,9% · Sedang 21,4% · Tinggi 80,5% |

---

## ✅ Conclusion

1. **Tingkat dropout tinggi.** Sebanyak **32,1% mahasiswa (1.421 dari 4.424) dropout**, sekitar 1 dari 3 mahasiswa. Sebanyak 17,9% masih terdaftar setelah masa studi normal (terlambat lulus) dan 49,9% lulus.

2. **Faktor yang paling berkaitan dengan dropout:**

   | Faktor | Temuan |
   |---|---|
   | **Akademik** (paling kuat) | **84%** mahasiswa yang tidak lulus satu pun MK di semester 2 akhirnya dropout, vs 11% yang lulus 5–6 MK. Rata-rata mahasiswa dropout hanya lulus 2,6 MK (smt 1) dan 1,9 MK (smt 2), sedangkan mahasiswa yang lulus sekitar 6 MK. |
   | **Finansial** | Menunggak UKT: **87%** dropout (vs 25% yang lunas). Memiliki utang: **62%** (vs 28%). Penerima beasiswa hanya **12%** (vs 39%). |
   | **Demografis** | Usia masuk ≥ 25 tahun: **> 50%** (vs 21% usia ≤ 20). Laki-laki 45% (vs 25% perempuan). Kelas malam 43% (vs 31% kelas siang). |
   | **Prodi & jalur masuk** | Tertinggi: Equinculture (55%), Informatics Engineering (54%), Management kelas malam (51%). Terendah: Nursing (15%). Jalur *Over 23 years old* (55%) dan *Holders of other higher courses* (61%) paling berisiko. |
   | **Tidak berpengaruh berarti** | Pendidikan orang tua dan kondisi makroekonomi (pengangguran, inflasi, GDP) |

3. **Deteksi dini dapat dilakukan otomatis.** Model mendeteksi **81% mahasiswa yang akan dropout** dengan precision 80%. Tingkat risikonya terbukti bermakna: dropout rate aktual 4,9% (Rendah), 21,4% (Sedang), dan 80,5% (Tinggi).

4. **Monitoring berkelanjutan** dapat dilakukan lewat dashboard Metabase dan tab Dashboard di EduGuard, per program studi, gender, dan waktu kuliah.

### 🎯 Rekomendasi Action Items

| # | Aksi | Detail |
|---|---|---|
| 1 | **Peringatan dini tiap akhir semester** | Jalankan prediksi batch EduGuard untuk seluruh mahasiswa aktif di akhir semester 1 dan 2. Mahasiswa risiko *Tinggi* bertemu dosen wali dalam 2 minggu; risiko *Sedang* dipantau bulanan. |
| 2 | **Intervensi akademik sejak semester 1** | Bimbingan intensif, kelas remedial, dan tutor sebaya bagi mahasiswa dengan rasio kelulusan MK < 50% atau tanpa MK lulus. |
| 3 | **Tangani masalah finansial lebih awal** | Integrasikan data tunggakan UKT ke sistem akademik; tawarkan cicilan/keringanan UKT dan konseling finansial bagi yang menunggak atau berutang. |
| 4 | **Perluas beasiswa** | Prioritaskan mahasiswa berisiko tinggi dengan kendala finansial, karena dropout rate penerima beasiswa jauh lebih rendah. |
| 5 | **Dukung mahasiswa dewasa & kelas malam** | Jadwal fleksibel, opsi kelas daring/hybrid, dan konseling manajemen waktu bagi mahasiswa ≥ 25 tahun, jalur *Over 23 years old*, dan kelas malam. |
| 6 | **Mentoring prodi berisiko tinggi** | Program mentoring khusus dan evaluasi kurikulum tahun pertama untuk Equinculture, Informatics Engineering, dan Management kelas malam. |
| 7 | **Monitoring & evaluasi berkala** | Tinjau dashboard setiap semester untuk melihat dampak intervensi, dan latih ulang model setiap tahun dengan data terbaru. |

---

## 📋 Checklist Submission

**Kriteria wajib**
- [x] Menggunakan template proyek (`notebook.ipynb`, `README.md`)
- [x] Proses data science lengkap: business understanding → data understanding → preparation → modeling → evaluation → deployment
- [x] Business dashboard Metabase + `metabase.db.mv.db` + kredensial akses
- [x] Prototype machine learning dengan Streamlit, di-*deploy* ke Streamlit Community Cloud
- [x] Kesimpulan dan rekomendasi action items

**Saran**
- [x] Dokumentasi setiap tahapan dengan *text cell* di notebook, termasuk insight dari setiap analisis
- [x] Visualisasi data yang efektif: palet warna konsisten dan aman untuk buta warna, sumbu dimulai dari nol, label nilai langsung pada grafik
- [x] Prototype dengan tampilan UI yang rapi dan mudah digunakan
- [ ] Video penjelasan (maksimal 5 menit)

---

## 📄 Sumber Data

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

*Jaya Jaya Institut adalah nama fiktif yang digunakan untuk keperluan proyek pembelajaran Dicoding.*
