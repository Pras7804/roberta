# 🧬 BIOPROSES SEL — Penilaian Esai Biologi dengan Bi-Encoder + Cosine Similarity

Aplikasi Streamlit untuk guru Biologi SMA kelas XI, materi **Transpor Membran**. Hanya menggunakan **XLM-RoBERTa-base Bi-Encoder + Cosine Similarity**, tanpa regresi linear maupun Cross-Encoder.

## 1. Isi folder utama

```text
RISET_XLM-ROBERTA/
├── riset.py                    # HANYA aplikasi Bi-Encoder + cosine
├── model_aes.py                # prapemrosesan, bank soal, cosine, rubrik
├── hasil_split.xlsx            # dataset Anda: 37 kode soal, 1.258 jawaban
├── kamus_colab.json            # 5.274 pasangan normalisasi dari Colab
├── cover_lkpd.png              # cover LKPD asli pada desain header
├── requirements.txt           # pustaka dibutuhkan
├── README.md
├── MULAI_DISINI.txt
├── .streamlit/
│   └── config.toml            # tema cerah
└── tests/
    └── test_model_aes.py      # pemeriksaan perhitungan ringan
```

Folder baru ini **tidak membutuhkan** `model_cross`, `cross_encoder.pt`, `metadata.json`, `model_bi_ols.json`, maupun `latih_cross_encoder.py`. Supaya benar-benar bersih, buat folder proyek baru dari ZIP, jangan campur dengan file lama.

## 2. Cara menjalankan di Visual Studio Code (Windows)

1. Ekstrak semua ZIP ke folder baru `RISET_XLM-ROBERTA-BI`.
2. Buka **File → Open Folder** di VS Code dan pilih folder tersebut.
3. Buka **Terminal → New Terminal** dan jalankan:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run riset.py
```

4. Buka alamat yang tertera pada terminal, biasanya `http://localhost:8501`.
5. Jika ada Streamlit lama yang memakai port 8501, hentikan dulu dengan `Ctrl+C` pada terminalnya, lalu jalankan perintah di atas.
6. Pertama kali digunakan, Transformers perlu mengunduh `xlm-roberta-base` dari Hugging Face (model relatif besar); selanjutnya di-cache otomatis.

## 3. Fitur

- **Pilih kode soal 1 sampai 37**, seperti antarmuka teman Anda. Pertanyaan, jawaban referensi guru, dan bobot diambil otomatis dari Excel asli.
- **Koreksi satu jawaban**: guru mengetik teks baru atau memilih jawaban dari dataset.
- **Koreksi semua 34 siswa untuk satu kode soal**: hasil per siswa muncul dalam tabel.
- **Unggah file Excel/CSV**: sistem menerima kolom `Jawaban_Siswa` (wajib), `Kode_Soal`, `Nama`, `Kelas`, `Absen` (opsional). Kolom kode tidak ada → menggunakan kode terpilih; kode ada → bisa multi-soal sekaligus.
- **Ekspor hasil Excel**: skor cosine, persentase kemiripan, skor kontinu, skor rubrik dan selisih dari nilai guru (jika tersedia).
- **Progress sesi** dihitung berdasarkan kode soal yang sudah dinilai dalam sesi, bukan data yang sudah dikoreksi permanen.
- **Tampilan cerah** dari cover LKPD; gradasi biru, cyan, kuning, hijau. Tidak mengubah konsep materi atau soal.

## 4. Kesetaraan dengan kode Bi-Encoder Colab TANPA regresi

1. Prapemrosesan `lowercase → whitespace → hapus tanda baca → kamus` menggunakan `kamus_colab.json` yang diekspor dari notebook asli.
2. `AutoTokenizer.from_pretrained("xlm-roberta-base")` dan `XLMRobertaModel.from_pretrained`.
3. Dua teks **jawaban siswa dan kunci guru** diencode secara terpisah (tanpa pertanyaan pada input) dengan `padding="max_length"`, `max_length=512`, `truncation=True`.
4. Embedding = `last_hidden_state[:,0,:]` (768 dimensi); model `.eval()`, tidak dilakukan pembaruan bobot.
5. `sklearn.metrics.pairwise.cosine_similarity` dibatasi `[0,1]`.
6. Skor kontinu = `cosine * Bobot`.
7. Skor rubrik = `max(0, np.round(skor_kontinu*2)/2)`; pembulatan kelipatan 0,5 sama dengan notebook.

**Catatan:** tanpa akses pada runtime Colab yang sama, angka hasil inferensi belum diverifikasi identik baris-per-baris. Perbedaan versi PyTorch/Transformers, CPU/GPU atau precision bisa menyebabkan selisih sangat kecil. Skor cosine menunjukkan kemiripan representasi, **bukan akurasi model dan bukan keputusan mutlak kualitas ilmiah jawaban siswa**.

## 5. Catatan dataset dan privasi

Dataset sumber memuat **34 siswa × 37 soal** (total 1.258 jawaban), beserta nama siswa; gunakan secara lokal dan jangan membagikan identitas tanpa izin. Soal yang merujuk ke gambar pada LKPD dapat tetap memerlukan pengamatan gambar secara manual oleh guru. Aplikasi tidak menafsirkan gambar soal melalui encoder teks.

## 6. Uji tanpa mengunduh model

```powershell
python -m unittest discover -s tests -v
```
