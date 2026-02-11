"""Tests for the Predictor wrapper."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from app.inference.predictor import Predictor


@pytest.fixture
def mock_model():
    model = MagicMock()
    # predict_proba returns [[p0, p1], ...] — we use column 1
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4]])
    return model


class TestPredictor:
    def test_returns_class1_probabilities(self, mock_model):
        predictor = Predictor(mock_model)
        result = predictor.predict(np.array([[1, 2], [3, 4]], dtype=np.float32))
        np.testing.assert_array_almost_equal(result, [0.8, 0.4])

    def test_rejects_1d_input(self, mock_model):
        predictor = Predictor(mock_model)
        with pytest.raises(ValueError, match="batch"):
            predictor.predict(np.array([1.0, 2.0]))

    def test_calls_model_once(self, mock_model):
        predictor = Predictor(mock_model)
        batch = np.array([[1, 2]], dtype=np.float32)
        predictor.predict(batch)
        mock_model.predict_proba.assert_called_once()
