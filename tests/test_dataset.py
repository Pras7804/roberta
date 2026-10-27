"""Pastikan pemilihan soal benar-benar memakai 37 kode pada data pengguna."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_aes import siap_dataset,bank_dari_dataset

class UjiBankSoal(unittest.TestCase):
    def test_kode_dan_jawaban(self):
        path = Path(__file__).resolve().parents[1] / "hasil_split.xlsx"
        df = siap_dataset(pd.read_excel(path, sheet_name="Semua_Data"))
        bank = bank_dari_dataset(df)
        self.assertEqual(len(df), 1258)
        self.assertEqual(sorted(bank), list(range(1, 38)))
        self.assertTrue(all(v["jumlah_jawaban"] == 34 for v in bank.values()))
        self.assertEqual(bank[5]["bobot"], 0.5)

if __name__ == "__main__":
    unittest.main()
