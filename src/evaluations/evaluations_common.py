from common.settings import LLMConfig, Settings
from evaluations.inference_prompter import Prompter
from evaluations.llm_as_a_judge import LLMAsAJudge


def get_inference_model(settings: Settings, conf_elem: dict, prompter: Prompter):
    if conf_elem.get("llm_config", "") != "":
        return LLMAsAJudge(
            settings=settings,
            llm_config=LLMConfig(**conf_elem["llm_config"]),
            prompter=prompter,
        )

    elif conf_elem.get("slm_model", "") != "":
        from evaluations.slm_as_a_judge import TransformerSLM

        return TransformerSLM(
            model_name=conf_elem["slm_model"],
            prompter=prompter,
        )

    else:
        raise Exception("Missing inference model in configuration file")
