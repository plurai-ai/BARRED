from pathlib import Path

import pandas as pd

from common.common_types import TaskDefinition
from data_handler.data_class import SampleDataset, TaskAndResponseClsfSample


INSTRUCTION_TO_REMOVE = "Beware that you have {remaining_steps} steps remaining."


class PlanVerificationDataset(SampleDataset):
    def _postprocess_samples(self) -> None:
        ann_df = self.test_samples
        ann_df_pos, ann_df_neg = ann_df.copy(deep=True), ann_df.copy(deep=True)

        ann_df_pos["model_response"] = ann_df_pos["original_task_output"]
        ann_df_pos["predicate_label"] = "PASS"

        ann_df_neg["model_response"] = ann_df_neg["violating_task_output"]
        ann_df_neg["predicate_label"] = "FAIL"

        self.test_samples = pd.concat([ann_df_pos, ann_df_neg])
        self.test_samples.drop(columns=["original_task_output", "violating_task_output"], inplace=True)

    def load_test_file(self, test_file: str) -> None:  # type: ignore[override]
        self.test_name = Path(test_file).stem
        self.test_samples = pd.read_csv(test_file)
        self._postprocess_samples()

    def parse_sample(self, sample):
        return sample["task_input"], sample["model_response"]

    def create_clsf_sample_from_test_sample(
        self, sample: dict, task_def: TaskDefinition
    ):
        task_input, task_output = self.parse_sample(sample)
        task_input = task_input.replace(INSTRUCTION_TO_REMOVE, "")
        return TaskAndResponseClsfSample(
            task_input=task_input,
            model_response=task_output,
            rule=task_def.cluster,
            labels=task_def.labels,
        )
