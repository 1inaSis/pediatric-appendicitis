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
    assert arr.mean() == 2.0