from typing import Optional, Union

import torch
from tuned_lens import TunedLens
from tuned_lens.nn.lenses import LogitLens
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM
"""
ADAPTED FROM https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/mamba_lm.py
"""

@register_model("pythia_lens")
class PythiaLens(HFLM):
    VALID_SIZES = ['70m', '160m', '410m', '1.4b', '2.8b', '8B']
    def __init__(
        self,
        model_size='410m',
        evaluation_layer=-1,
        lens_type='tuned',
        model_name='Pythia'
    ) -> None:
        assert model_size in self.VALID_SIZES
        if model_name == 'Pythia':
            model_path=f"EleutherAI/pythia-{model_size}-deduped"
        elif model_name == 'Llama3':
            model_path="meta-llama/Meta-Llama-3-8B"
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        self.is_hf = True
        self.evaluation_layer = evaluation_layer
        self.lens_type = lens_type
        super().__init__(
            pretrained=model_path,
            tokenizer=model_path,
            max_length=2048,
        )

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        **kwargs,
    ) -> None:

        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map='auto' if torch.cuda.is_available() else None,
            revision='main'
        )

        # If no CUDA available, manually move to appropriate device
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self._model = self._model.to('mps')
            else:
                self._model = self._model.to('cpu')

        print(self.evaluation_layer, self.config.num_hidden_layers)
        if self.lens_type == 'tuned':
            self.lens = TunedLens.from_model_and_pretrained(self._model)
        else:
            self.lens = LogitLens.from_model(self._model)
        self.lens.to(self._model.device)

        assert self.evaluation_layer <= self.config.num_hidden_layers, \
            f"Evaluation layer={self.evaluation_layer} cannot be larger than the number of layers={self.config.num_hidden_layers}! "

    def _model_call(self, inps, attn_mask=None, labels=None):
        # returns the logits
        assert self.AUTO_MODEL_CLASS == AutoModelForCausalLM

        outputs = self._model(inps, output_hidden_states=True)

        hs = list(outputs.hidden_states)
        logits = self.lens.forward(hs[self.evaluation_layer], self.evaluation_layer )
        return logits
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        generation_kwargs["do_sample"] = generation_kwargs.get("do_sample", False)

        return self.lens.generate(
            model=self._model,
            layer=self.evaluation_layer,
            input_ids=context,
            max_new_tokens=5,
            temp = generation_kwargs["temperature"],
            do_sample = generation_kwargs["do_sample"]
        )
