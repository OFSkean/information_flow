import logging
import time
from typing import Any, Callable, List, Literal, Type, Dict, Union
import gc

import tqdm
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ..misc.model_dataloader_utils import get_model_path, model_name_to_sizes
from ..misc.batch_size_utils import find_optimal_batch_size
from llm2vec import LLM2Vec


class ModelSpecifications:
    def __init__(self, model_family, model_size, revision):
        self.model_family = model_family
        self.model_size = model_size
        self.revision = revision

        self.do_checks()
    
    def do_checks(self):
        if self.revision != "main":
            # currently only supporting 14m and 410m Pythia models for non-main checkpoints
            assert self.model_family == "Pythia"
            assert self.model_size in ["14m", "410m"]

        assert self.model_family in model_name_to_sizes.keys()
        assert self.model_size in model_name_to_sizes[self.model_family], \
            f"Model size {self.model_size} not found for model family {self.model_family}, available sizes: {model_name_to_sizes[self.model_family]}"

class AutoModelWrapper:
    def __init__(self, model_specs: ModelSpecifications, device_map="auto", evaluation_layer_idx: int = -1):
        model_path = get_model_path(model_specs.model_family, model_specs.model_size)
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        assert self.tokenizer.pad_token is not None

        self.config = AutoConfig.from_pretrained(model_path, 
                                                 revision=model_specs.revision,
                                                 output_hidden_states=True)
        self.num_layers = self.config.num_hidden_layers + 1 
        self.update_evaluation_layer(evaluation_layer_idx)
        self.config.num_hidden_layers = self.evaluation_layer_idx

        FROM_PRETRAINED_KWARGS = {
            'revision': model_specs.revision,
            'config': self.config,
            'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            'device_map': device_map
        }

        if 'llm2vec' in model_path.lower():
            MODEL_CLASS = LLM2Vec
            if 'unsup' in model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
            elif 'supervised' in model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
            elif model_specs.model_family.lower() == 'llm2vec-mntp':
                pass
            else:
                raise ValueError(f"Model family {model_specs.model_family} not found")
        else:
            MODEL_CLASS = AutoModel

        self.model = MODEL_CLASS.from_pretrained(model_path, **FROM_PRETRAINED_KWARGS).eval()

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

    def update_evaluation_layer(self, evaluation_layer_idx):
        if evaluation_layer_idx == -1:
            self.evaluation_layer_idx = self.num_layers - 1
        else:
            self.evaluation_layer_idx = evaluation_layer_idx

        assert self.evaluation_layer_idx >= 0 and self.evaluation_layer_idx < self.num_layers, \
            f"Evaluation layer {self.evaluation_layer_idx} is not in the range of the model's hidden layers 0 to {self.num_layers - 1}"
    
    @torch.no_grad()
    def encode(
        self,
        sentences: List[str],
        **kwargs: Any
    ) -> np.ndarray:
        max_sample_length = kwargs.pop("max_sample_length", 2048)
        verbose = kwargs.pop("verbose", True)

        tokenized_sentences =  self.tokenizer(sentences,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=max_sample_length)
        
        # find optimal batch size
        optimal_batch_size = find_optimal_batch_size(model=self._get_model_with_forward_pass(), 
                                                     number_of_samples=len(sentences),
                                                     device=self._get_first_layer_device(),
                                                     max_sentence_length = tokenized_sentences.input_ids.shape[1], 
                                                     verbose=verbose)
        self.batch_size_hint = optimal_batch_size

        # create dataloader
        dataset = [{"input_ids": ids, "attention_mask": mask} 
            for ids, mask in zip(tokenized_sentences["input_ids"], 
                                tokenized_sentences["attention_mask"])]
        dataloader = DataLoader(dataset, 
                                batch_size=optimal_batch_size, 
                                shuffle=False, 
                                num_workers=8, 
                                collate_fn=self.collate)

        embeddings = self._encode(dataloader, verbose=verbose)

        return np.array(embeddings)

    def _get_hf_device_map(self):
        if hasattr(self.model, 'hf_device_map'):
            return self.model.hf_device_map
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'hf_device_map'):
            return self.model.model.hf_device_map
        else:
            raise ValueError("Could not find hf_device_map")
    
    def _get_first_layer_device(self):
        device_map = self._get_hf_device_map()
        first_layer_name = list(device_map.keys())[0]
        return device_map[first_layer_name]
    
    def _get_model_with_forward_pass(self):
        if 'llm2vec' in self.model_path.lower():
            return self.model.model
        else:
            return self.model
    
    @property
    def device(self):
        return self._get_first_layer_device()
    
    def forward(self, **kwargs):
        model_with_forward_pass = self._get_model_with_forward_pass()
        return model_with_forward_pass(**kwargs)
    
    def __call__(self, **kwargs):
        return self.forward(**kwargs)
    
    @torch.no_grad()
    def _encode(self, dataloader, verbose=False) -> np.ndarray:
        encoded_batches = []

        for batch in tqdm.tqdm(dataloader, total=len(dataloader), disable= not verbose):
            batch = {k: v.to(self._get_first_layer_device()) for k, v in batch.items()}
                
            outputs = self.forward(**batch)
            hidden_states = outputs.hidden_states[self.evaluation_layer_idx]
            hidden_states = self._get_pooled_hidden_states(hidden_states, batch["attention_mask"], method="mean")

            encoded_batches.append(hidden_states.float().cpu())

        encodings = torch.cat(encoded_batches).squeeze().numpy()
        return encodings
    
    @torch.no_grad()
    def _get_pooled_hidden_states(self, hidden_states, attention_mask, method="mean"):
        if method == "mean":
            seq_lengths = attention_mask.sum(dim=-1)
            return torch.stack(
                [
                    hidden_states[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif method == "mean_including_padding":
            layer_means = torch.stack([torch.mean(x, dim=0) for x in hidden_states])
            return layer_means
        
        elif method == "last_hidden_state":
            return hidden_states[:, -1]
        else:
            raise ValueError(f"Invalid pooling method: {method}")

    def collate(self, batch):
        ips = [item['input_ids'] for item in batch]
        attn = [item['attention_mask'] for item in batch]

        return {'input_ids': torch.stack(ips),
                'attention_mask': torch.stack(attn),
                }
