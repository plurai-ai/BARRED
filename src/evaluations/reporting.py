from typing import List

import numpy as np
from pydantic import BaseModel


class Prediction(BaseModel):
    sample: str
    gt_label: str
    predicted_label: str
    latency: float

    def is_correct(self) -> bool:
        return self.predicted_label.lower() == self.gt_label.lower()


class ModelPerformance(BaseModel):
    predictions: List[Prediction]
    accuracy: float
    precision: float
    recall: float
    avg_latency: float
    convergence_history: list[float] | None = None

    @staticmethod
    def from_model_results(
        labels: list[str],
        num_correct: dict[str, int],
        num_predicted: dict[str, int],
        num_samples: dict[str, int],
        predictions: List[Prediction],
    ):
        recall_per_label, precision_per_label = {}, {}

        for label in labels:
            recall_per_label[label] = (
                num_correct.get(label, 0) / num_samples[label]
                if label in num_samples
                else 0.0
            )
            precision_per_label[label] = (
                num_correct.get(label, 0) / num_predicted[label]
                if label in num_predicted
                else 0.0
            )

        accuracy = sum(num_correct.values()) / sum(num_samples.values())

        return ModelPerformance(
            predictions=predictions,
            accuracy=accuracy,
            precision=float(np.mean(list(precision_per_label.values()))),
            recall=float(np.mean(list(recall_per_label.values()))),
            avg_latency=float(
                np.mean([prediction_elem.latency for prediction_elem in predictions])
            ),
        )
