import logging
from typing import Any, Callable, List, Literal, Type, Dict, Union

import numpy as np
import torch

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta

from mteb.models.instructions import task_to_instruction
from models.llm2vec import LLM2Vec

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


class LLM2VecWrapper:
    def __init__(self, *args, **kwargs):
        extra_kwargs = {}
        try:
            import flash_attn

            extra_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning(
                "LLM2Vec models were trained with flash attention enabled. For optimal performance, please install the `flash_attn` package with `pip install flash-attn --no-build-isolation`."
            )
        self.task_to_instructions = None
        if "task_to_instructions" in kwargs:
            self.task_to_instructions = kwargs.pop("task_to_instructions")

        self.model = LLM2Vec.from_pretrained(*args, **extra_kwargs, **kwargs)

    def encode(
        self,
        sentences: List[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if prompt_name is not None:
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                and prompt_name in self.task_to_instructions
                else task_to_instruction(prompt_name)
            )
        else:
            instruction = ""

        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List[str]], List[str]],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sep = " "
        if isinstance(corpus, Dict):
            sentences = [
                (corpus["title"][i] + sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            if isinstance(corpus[0], str):
                sentences = corpus
            else:
                sentences = [
                    (doc["title"] + sep + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                    for doc in corpus
                ]
        sentences = [["", sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)


def _loader(wrapper: Type[LLM2VecWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner

llm2vec_llama3_8b_supervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

llm2vec_llama3_8b_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

baseline_llama3 = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="meta-llama/Meta-Llama-3-8B",
        peft_model_name_or_path=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="meta-llama/Meta-Llama-3-8B",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

unidirectional_llama3 = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="meta-llama/Meta-Llama-3-8B",
        peft_model_name_or_path=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        enable_bidirectional=False,
    ),
    name="meta-llama/Meta-Llama-3-8B",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

def convert_model_name_to_loader(model_name: str):
    model_name_to_model = {
        'unidirectional-llama': unidirectional_llama3,
        'baseline-llama': baseline_llama3,
        'llm2vec-unsupervised': llm2vec_llama3_8b_unsupervised,
        'llm2vec-supervised': llm2vec_llama3_8b_supervised,
        
    }

    if model_name in model_name_to_model:
        return model_name_to_model[model_name]
    else:
        raise ValueError(f"Unknown model name: {model_name}, valid choices are {model_name_to_model.keys()}")