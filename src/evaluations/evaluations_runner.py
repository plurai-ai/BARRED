import argparse
import logging
from pathlib import Path

import yaml

from common.common_types import TaskDefinition
from common.prompt_utils import render_jinja_prompts
from data_handler.data_class import SampleDataset
from common.settings import Settings
from evaluations.evaluations_common import get_inference_model
from evaluations.inference_prompter import Prompter


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_config_file", type=str, required=True, help="Path to the test config YAML file")
    args = parser.parse_args()

    test_config_file = args.test_config_file
    with open(test_config_file) as f:
        test_config = yaml.safe_load(f)

    classification_type = test_config["classification_type"]
    dataset_conf = test_config["dataset"]

    prompts_file = str(Path(__file__).parent.parent.parent / "prompts/prompts.yaml.jinja2")
    prompts_data = render_jinja_prompts(prompts_file, {"clsf_type": classification_type})
    prompter = Prompter(prompts_data)

    settings = Settings()
    inference_model = get_inference_model(settings, dataset_conf["evaluator"], prompter)
    task_def = TaskDefinition(cluster=dataset_conf["cluster"], labels=dataset_conf["labels"])

    test_dataset = SampleDataset.create(dataset_conf["name"])
    if "hf_dataset" in dataset_conf:
        test_dataset.load_hf_dataset(
            dataset_conf["hf_dataset"],
            dataset_conf.get("hf_config", dataset_conf["name"]),
            dataset_conf.get("hf_split", "test"),
        )
    else:
        test_dataset.load_test_file(test_file=dataset_conf["test_file"])
    inference_model.run(test_dataset, task_def, "Main test set", test_config["out_dir"])
