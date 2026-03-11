import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import optimize_memory


class TestMemoryOptimization(unittest.TestCase):

    def test_memory_reduction(self):
        """
        Ce test vérifie que la fonction optimize_memory
        réduit (ou au minimum n'augmente pas) l'utilisation de la mémoire.
        """

        # Création d'un petit DataFrame de test
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],           # colonne entière
            "b": [1.1, 2.2, 3.3, 4.4, 5.5]  # colonne flottante
        })

        # Calcul de la mémoire avant optimisation
        mem_before = df.memory_usage().sum()

        # Application de la fonction
        df_optimized = optimize_memory(df)

        # Calcul de la mémoire après optimisation
        mem_after = df_optimized.memory_usage().sum()

        # Vérifier que la mémoire n'a pas augmenté
        self.assertTrue(mem_after <= mem_before)


# Permet d'exécuter les tests si le fichier est lancé directement
if __name__ == "__main__":
    unittest.main()