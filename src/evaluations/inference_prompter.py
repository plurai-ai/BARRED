from typing import Any

from data_handler.data_class import ClsfSample, TaskAndResponseClsfSample, InputBlockClsfSample


class Prompter:
    def __init__(self, prompts_data: dict[str, Any]):
        self.system_msg = prompts_data["llm_as_a_judge"]["system"].strip()
        self.user_msg = prompts_data["llm_as_a_judge"]["human"].strip()

        self.no_reasoning_instructions = prompts_data["llm_as_a_judge"]["no_reasoning_instructions"]

    def get_system_msg(self, clsf_sample: ClsfSample) -> str:
        return self.system_msg.format(
            rule=clsf_sample.rule,
            possible_labels=clsf_sample.labels,
            output_format=self.no_reasoning_instructions,
        )

    def get_user_msg(self, clsf_sample: ClsfSample) -> str:
        if isinstance(clsf_sample, TaskAndResponseClsfSample):
            return self.user_msg.format(
                task_definition=clsf_sample.task_input,
                model_response=clsf_sample.model_response
            )
        elif isinstance(clsf_sample, InputBlockClsfSample):
            return self.user_msg.format(input_block=clsf_sample.input_block)
        else:
            raise Exception("Invalid classification type")
