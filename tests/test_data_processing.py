import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('.')


def test_basic():
    """Test basique — vérifie que pytest fonctionne."""
    assert 1 + 1 == 2


def test_dataframe_creation():
    """Test que pandas fonctionne."""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert df.shape == (2, 2)


def test_numpy_works():
    """Test que numpy fonctionne."""
    arr = np.array([1, 2, 3])
    assert arr.mean() == 2

import unittest

data_processed_and_balanced= pd.read_excel("data/data_processed_and_balanced.xlsxp")

class TestDataProcessing(unittest.TestCase):

    def test_dataset_not_empty(self):
        """Vérifie que le dataset traité n'est pas vide"""
        self.assertTrue(len(data_processed_and_balanced) > 0)

    def test_target_exists(self):
        """Vérifie que la colonne cible existe"""
        self.assertIn("Diagnosis", data_processed_and_balanced.columns)

    def test_dataset_balanced(self):
        """Vérifie que les classes sont équilibrées"""
        counts = data_processed_and_balanced["Diagnosis"].value_counts()
        self.assertEqual(counts.iloc[0], counts.iloc[1])

    

if __name__ == "__main__":
    unittest.main()