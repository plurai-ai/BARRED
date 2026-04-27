from data_handler.data_class import ClsfSample
from evaluations.inference_prompter import Prompter
from evaluations.model_runner import InferenceModel


RESPONSE_TEMPLATE = "### Response"


class TransformerSLM(InferenceModel):
    MAX_SEQ_LENGTH = 5000

    def __init__(self, model_name: str, prompter: Prompter):
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=TransformerSLM.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,
        )

        FastLanguageModel.for_inference(self.model)

        self.model_name = model_name
        self.prompter = prompter

    def predict(self, clsf_sample: ClsfSample):
        system_msg = self.prompter.get_system_msg(clsf_sample)
        user_msg = self.prompter.get_user_msg(clsf_sample)
        input_to_guardrail_model = f"{system_msg}\n\n{user_msg}\n\n{RESPONSE_TEMPLATE}"

        input_tokens = self.tokenizer([input_to_guardrail_model], return_tensors="pt").to("cuda")
        output_tokens = self.model.generate(**input_tokens, max_new_tokens=500, use_cache=True)
        output_of_guardrail_model = self.tokenizer.batch_decode(output_tokens)[0]

        prediction_index = output_of_guardrail_model.rfind(RESPONSE_TEMPLATE)
        assert prediction_index != -1
        prediction_index += len(RESPONSE_TEMPLATE)

        prediction_suffix = output_of_guardrail_model[prediction_index:]
        prediction = prediction_suffix.strip()
        if prediction.startswith(":"):
            prediction = prediction[1:].strip()
        if prediction.endswith(self.tokenizer.eos_token):
            prediction = prediction[:-len(self.tokenizer.eos_token)]
        return prediction

    def get_model_name(self) -> str:
        return self.model_name
