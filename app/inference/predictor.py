from __future__ import annotations

from typing import Any

import numpy as np


class Predictor:
	"""Thin wrapper hiding the underlying model implementation."""

	def __init__(self, model: Any) -> None:
		self._model = model

	def predict(self, batch_inputs: np.ndarray) -> np.ndarray:
		if batch_inputs.ndim != 2:
			raise ValueError("Inputs must be shaped (batch, features)")
		probabilities = self._model.predict_proba(batch_inputs)
		return np.asarray(probabilities[:, 1], dtype=float)
