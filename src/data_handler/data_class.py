from typing import Any

from pydantic import BaseModel

from common.common_types import TaskDefinition


class SampleDataset:
    @staticmethod
    def create(name: str) -> "SampleDataset":
        if name == "message_repetition":
            from data_handler.message_repetition_dh import MessageRepetitionDataset
            return MessageRepetitionDataset()

        elif name == "gps_disclosure":
            from data_handler.gps_disclosure_dh import GpsDisclosureDataset
            return GpsDisclosureDataset()

        elif name == "plan_verification":
            from data_handler.plan_verification_dh import PlanVerificationDataset
            return PlanVerificationDataset()

        elif name == "healthe":
            from data_handler.healthe_dh import HealtheDataset
            return HealtheDataset()

        else:
            raise ValueError("Invalid dataset name")

    def load_test_file(self, test_file: str) -> None:
        pass

    def load_hf_dataset(self, hf_dataset: str, config_name: str, split: str = "test") -> None:
        from datasets import load_dataset
        ds = load_dataset(hf_dataset, name=config_name, split=split)
        self.test_name = config_name
        self.test_samples = ds.to_pandas()
        self._postprocess_samples()

    def _postprocess_samples(self) -> None:
        pass

    def get_test_samples(self) -> Any:
        return self.test_samples

    def load_custom_test(self, test_df: Any, test_name: str) -> None:
        self.test_name = test_name
        self.test_samples = test_df

    def parse_sample(self, sample: Any) -> Any:
        pass

    def create_clsf_sample_from_test_sample(
        self, sample: dict, task_def: TaskDefinition
    ) -> Any:
        pass


class ClsfSample(BaseModel):
    rule: str
    labels: list[str]


class TaskAndResponseClsfSample(ClsfSample):
    task_input: str
    model_response: str

    def get_sample_key(self):
        return f"{self.task_input}###{self.model_response}###{self.rule}"

    def log(self) -> str:
        return f"Task input: {self.task_input}\n\nModel response: {self.model_response}"


class InputBlockClsfSample(ClsfSample):
    input_block: str

    def get_sample_key(self):
        return f"{self.input_block}###{self.rule}"

    def log(self) -> str:
        return self.input_block
