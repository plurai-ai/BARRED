from langchain_core.messages import HumanMessage, SystemMessage

from common.llm import get_llm
from common.settings import LLMConfig, Settings
from data_handler.data_class import ClsfSample
from evaluations.inference_prompter import Prompter
from evaluations.model_runner import InferenceModel


class LLMAsAJudge(InferenceModel):
    def __init__(
        self, settings: Settings, llm_config: LLMConfig, prompter: Prompter
    ):
        self.llm = get_llm(settings, llm_config)
        self.llm_config = llm_config
        self.prompter = prompter

    def predict(self, clsf_sample: ClsfSample):
        messages = [
            SystemMessage(content=self.prompter.get_system_msg(clsf_sample)),
            HumanMessage(content=self.prompter.get_user_msg(clsf_sample)),
        ]

        response = self.llm.invoke(messages)
        llm_output = response.content.strip()
        return llm_output

    def get_model_name(self) -> str:
        return self.llm_config.name
