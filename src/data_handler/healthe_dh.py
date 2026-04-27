from pathlib import Path

import pandas as pd

from common.common_types import TaskDefinition
from data_handler.data_class import SampleDataset, InputBlockClsfSample


class HealtheDataset(SampleDataset):
    def _postprocess_samples(self) -> None:
        self.test_samples["predicate_label"] = self.test_samples.apply(
            lambda row: "True" if row["predicate_label"] == 1 else "False", axis=1
        )

    def load_test_file(self, test_file: str) -> None:
        self.test_name = Path(test_file).stem
        self.test_samples = pd.read_csv(test_file)
        self._postprocess_samples()

    def parse_sample(self, sample):
        return sample["transcript"]

    def create_clsf_sample_from_test_sample(
        self, sample: dict, task_def: TaskDefinition
    ):
        return InputBlockClsfSample(
            input_block=self.parse_sample(sample),
            rule=task_def.cluster,
            labels=task_def.labels,
        )
