import json
import logging
import os
import time
import traceback
from typing import List

import pandas as pd
from tqdm import tqdm

from common.common_types import TaskDefinition
from data_handler.data_class import ClsfSample, SampleDataset

from evaluations.reporting import ModelPerformance, Prediction


logger = logging.getLogger(__name__)


class InferenceModel:
    def run(
        self,
        dataset: SampleDataset,
        task_def: TaskDefinition,
        test_name: str,
        out_dir=None,
    ) -> ModelPerformance:
        num_samples, num_correct, num_predicted = {}, {}, {}

        input_samples = dataset.get_test_samples()

        gt_col = "predicate_label"

        logger.info(f"Running inference on {test_name}: {str(task_def)}")

        if len(input_samples) == 0:
            return ModelPerformance(
                predictions=[],
                accuracy=0,
                precision=0,
                recall=0,
                avg_latency=0,
            )

        predictions: List[Prediction] = []

        for _, sample in tqdm(input_samples.iterrows()):
            try:
                clsf_sample = dataset.create_clsf_sample_from_test_sample(
                    sample, task_def
                )
            except Exception:
                error_msg = traceback.format_exc()
                logger.error(error_msg)
                continue

            start_time = time.time()
            pred = self.predict(clsf_sample)
            latency = time.time() - start_time

            gt = str(sample[gt_col]).strip()

            prediction_elem = Prediction(
                sample=sample["sample"] if "sample" in sample else json.dumps(dict(sample)),
                gt_label=gt,
                predicted_label=pred,
                latency=latency,
            )
            predictions.append(prediction_elem)

            num_samples[gt] = num_samples.get(gt, 0) + 1
            num_predicted[pred] = num_predicted.get(pred, 0) + 1

            if prediction_elem.is_correct():
                num_correct[gt] = num_correct.get(gt, 0) + 1
            else:
                logger.info(f"### predicted {pred}, ground truth {gt}\n{clsf_sample.log()}\n")

        logger.info(f"Cluster: {task_def.cluster}")
        logger.info(f"Num samples: {num_samples}, total: {sum(num_samples.values())}")
        logger.info(f"Correctly recalled: {num_correct}, total {sum(num_correct.values())}")
        logger.info("Correctly recalled (pct): %s", {lbl: num_correct.get(lbl, 0) / cnt for lbl, cnt in num_samples.items()})
        logger.info(f"Num predicted: {num_predicted}")

        metrics = ModelPerformance.from_model_results(
            task_def.labels, num_correct, num_predicted, num_samples, predictions
        )

        if out_dir:
            self.save_results(metrics, dataset, out_dir)

        return metrics

    def predict(self, clsf_sample: ClsfSample):
        raise NotImplementedError()

    def get_model_name(self) -> str:
        raise NotImplementedError()

    def save_results(
        self, metrics: ModelPerformance, dataset: SampleDataset, out_dir: str
    ):
        predictions_df = pd.DataFrame(
            [pred.model_dump() for pred in metrics.predictions]
        )
        model_name = self.get_model_name().replace("/", "_")
        pred_out_file = f"{out_dir}/{dataset.__class__.__name__}_{dataset.test_name}_{model_name}_results.csv"
        os.makedirs(out_dir, exist_ok=True)
        predictions_df.to_csv(pred_out_file, index=False)
