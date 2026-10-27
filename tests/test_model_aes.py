"""Tes ringan tanpa perlu mengunduh XLM-RoBERTa."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_aes import cosine_ke_nilai, pra_pemrosesan, similarity_rows, muat_kamus

class UjiPenilaian(unittest.TestCase):
    def test_pembulatan(self):
        self.assertEqual(cosine_ke_nilai(0.76, 2)["skor_rubrik"], 1.5)
        self.assertEqual(cosine_ke_nilai(0.99, 0.5)["skor_rubrik"], 0.5)
        self.assertEqual(cosine_ke_nilai(0.05, 0.5)["skor_rubrik"], 0)
        self.assertEqual(cosine_ke_nilai(3, 2)["skor_kontinu"], 2)
    def test_normalisasi_kamus(self):
        kamus = {"yg": "yang", "dipusi": "difusi"}
        self.assertEqual(pra_pemrosesan("DIPUSI yg cepat!", kamus), "difusi yang cepat")
        self.assertEqual(pra_pemrosesan(None, kamus), "")
    def test_cosine(self):
        a = np.array([[1, 0], [0, 1]], dtype=np.float32)
        b = np.array([[1, 0], [1, 0]], dtype=np.float32)
        np.testing.assert_allclose(similarity_rows(a,b), [1,0], rtol=1e-6)
    def test_kamus_terbaca(self):
        path = Path(__file__).resolve().parents[1] / "kamus_colab.json"
        self.assertGreater(len(muat_kamus(path)), 100)

if __name__ == "__main__":
    unittest.main()
