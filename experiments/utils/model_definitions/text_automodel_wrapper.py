from typing import Any, List

import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from llm2vec import LLM2Vec

from .base_automodel_wrapper import BaseModelSpecifications, BaseLayerwiseAutoModelWrapper
from ..misc.optimal_batch_size import find_optimal_batch_size
from ..dataloaders.text_dataloader import collate as text_collate

model_types = ["cerebras",
                "Pythia", 
                "mamba", 
                "mamba2", 
                "Medical-Llama3", 
                "Llama3", 
                "bert", 
                "roberta",
                "LLM2Vec-mntp-unsup-simcse", 
                "LLM2Vec-mntp-supervised",
                "LLM2Vec-mntp",
                "llama-instruct"]

cerebras_sizes = ['111M', '256M', '590M', '1.3B', '2.7B', '6.7B', '13B'] # '13b' also exists but doesnt fit in 24G for bfloat16
Pythia_sizes = ['14m', '70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b'] # '12b' also exists but doesnt fit in 24G for bfloat16
mamba_sizes = ['130m', '370m', '790m', '1.4b', '2.8b']
mamba2_sizes = ['130m', '370m', '780m', '1.3b', '2.7b']
bert_sizes = ['base', 'large']
medical_llama3_sizes = ['8B'] # its only 8B model
llama3_sizes = ['8B'] 
LLM2Vec_sizes = ['8B']
llama_instruct_sizes = ['8B']

model_name_to_sizes = {
    'Pythia': Pythia_sizes,
    'cerebras': cerebras_sizes,
    'mamba': mamba_sizes,
    'mamba2': mamba2_sizes,
    'Medical-Llama3': medical_llama3_sizes,
    'Llama3': llama3_sizes,
    'bert': bert_sizes,
    'roberta': bert_sizes,
    'LLM2Vec-mntp-unsup-simcse': LLM2Vec_sizes,
    'llama-instruct': llama_instruct_sizes,
    'LLM2Vec-mntp-supervised': LLM2Vec_sizes,
    'LLM2Vec-mntp': LLM2Vec_sizes,
}


def get_model_path(name, size):
    assert name in model_types
    if name == "cerebras":
        assert size in cerebras_sizes
        return f"cerebras/Cerebras-GPT-{size}"
    elif name == "Pythia":
        assert size in Pythia_sizes
        return f"EleutherAI/pythia-{size}"
    elif name == "Medical-Llama3":
        assert size in medical_llama3_sizes
        return f"ruslanmv/Medical-Llama3-8B"
    elif name == "Llama3":
        assert size in llama3_sizes
        return f"meta-llama/Meta-Llama-3-8B"
    elif name == "mamba":
        assert size in mamba_sizes
        return f"state-spaces/mamba-{size}-hf"
    elif name == "mamba2":
        assert size in mamba2_sizes
        return f"state-spaces/mamba2-{size}-hf" 
    elif name == "bert":
        assert size in bert_sizes
        return f"bert-{size}-uncased"
    elif name == 'roberta':
        assert size in bert_sizes
        return f"FacebookAI/roberta-{size}"
    elif name == 'LLM2Vec-mntp-unsup-simcse':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == 'LLM2Vec-mntp-supervised':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == 'LLM2Vec-mntp':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == "llama-instruct":
        assert size in llama_instruct_sizes
        return f"meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        raise ValueError(f"Model type {name} not found")




class TextModelSpecifications(BaseModelSpecifications):
    def __init__(self, model_family, model_size, revision, ignore_checks=False):
        super().__init__(model_family, model_size, revision, ignore_checks)
        self.model_path_func = get_model_path

    def additional_checks(self):
        if self.revision != "main":
            # currently only supporting 14m and 410m Pythia models for non-main checkpoints
            assert self.model_family == "Pythia"
            assert self.model_size in ["14m", "410m"]
        
        assert self.model_family in model_name_to_sizes.keys(), \
            f"Model family {self.model_family} not found, available families: {model_name_to_sizes.keys()}"
        assert self.model_size in model_name_to_sizes[self.model_family], \
            f"Model size {self.model_size} not found for model family {self.model_family}, available sizes: {model_name_to_sizes[self.model_family]}"

class TextLayerwiseAutoModelWrapper(BaseLayerwiseAutoModelWrapper):
    def __init__(self, 
                 model_specs: TextModelSpecifications, 
                 device_map="auto", 
                 evaluation_layer_idx: int = -1):
        super().__init__(model_specs, device_map, evaluation_layer_idx)

    """
    FUNCTIONS FOR INITIALIZATION
    """
    def setup_input_processor(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        assert self.tokenizer.pad_token is not None

        # number of tokens the model can handle
        self.max_tokens = self.tokenizer.model_max_length

    def setup_model(self):
        self.config = AutoConfig.from_pretrained(self.model_path, 
                                            revision=self.model_specs.revision,
                                            output_hidden_states=True)
        self.num_layers = self.config.num_hidden_layers + 1 
        self.update_evaluation_layer()
        self.config.num_hidden_layers = self.evaluation_layer_idx # prevents loading all layers

        FROM_PRETRAINED_KWARGS = {
            'revision': self.model_specs.revision,
            'config': self.config,
            'torch_dtype': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            'device_map': self.device_map
        }

        if 'llm2vec' in self.model_path.lower():
            MODEL_CLASS = LLM2Vec
            if 'unsup' in self.model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
            elif 'supervised' in self.model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
            elif self.model_specs.model_family.lower() == 'llm2vec-mntp':
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
            else:
                raise ValueError(f"Model family {self.model_specs.model_family} not found")
        else:
            MODEL_CLASS = AutoModelForCausalLM

        self.model = MODEL_CLASS.from_pretrained(self.model_path, **FROM_PRETRAINED_KWARGS).eval()        

    """
    FUNCTIONS FOR INFERENCE
    """
    @torch.no_grad()
    def encode(
        self,
        input_data: List[str],
        return_raw_hidden_states: bool = False,
        **kwargs: dict
    ) -> np.ndarray:
        max_sample_length = kwargs.pop("max_sample_length", 2048)
        if self.model_specs.model_family in ["bert", "roberta"]:
            max_sample_length = 512
            
        verbose = kwargs.pop("verbose", True)

        tokenized_sentences =  self.tokenizer(input_data,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=max_sample_length)
        
        # find optimal batch size
        optimal_batch_size = find_optimal_batch_size(model=self._get_model_with_forward_pass(), 
                                                     number_of_samples=len(input_data),
                                                     device=self.device,
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
                                collate_fn=text_collate)

        if return_raw_hidden_states:
            embeddings, raw_hidden_states, layerwise_encodings = self._encode_helper(dataloader, 
                                                            verbose=verbose, 
                                                            return_raw_hidden_states=return_raw_hidden_states,
                                                            **kwargs)
            return np.array(embeddings), raw_hidden_states, layerwise_encodings
        
        else:
            embeddings = self._encode_helper(dataloader, 
                                            verbose=verbose, 
                                            return_raw_hidden_states=return_raw_hidden_states,
                                            **kwargs) # shape: (num_samples, embedding_dim)
            return np.array(embeddings)
    
    
    def _get_model_with_forward_pass(self):
        if 'llm2vec' in self.model_path.lower():
            return self.model.model
        else:
            return self.model
    
    @torch.no_grad()
    def _encode_helper(self, dataloader, verbose=False, return_raw_hidden_states=False, **kwargs) -> np.ndarray:
        pooling_method = kwargs.pop("pooling_method", "mean")
        encoded_batches = []
        layerwise_encoded_batches = []

        if return_raw_hidden_states:
            # can be memory intensive, so only do if needed
            raw_sample_hidden_states = []

        for batch in tqdm.tqdm(dataloader, total=len(dataloader), disable= not verbose):
            batch = self.prepare_inputs(batch)
            
            outputs = self.forward(**batch)

            hidden_states = outputs.hidden_states[self.evaluation_layer_idx]
            
            hidden_states = self._get_pooled_hidden_states(hidden_states, batch["attention_mask"], method=pooling_method)
            encoded_batches.append(hidden_states.float().cpu())

            if return_raw_hidden_states:
                # get layerwise encodings for the batch
                current_batch_layerwise_encodings = []
                for layer_idx in range(len(outputs.hidden_states)):
                    layer_states = outputs.hidden_states[layer_idx]


                    layer_states = self._get_pooled_hidden_states(layer_states, batch["attention_mask"], method=pooling_method)
                    current_batch_layerwise_encodings.append(layer_states.float().cpu())
                layerwise_encoded_batches.append(torch.stack(current_batch_layerwise_encodings))
     
                # get raw hidden states for each sample
                for sample_idx in range(len(outputs.hidden_states[0])):
                    pad_idx = batch['attention_mask'][sample_idx] == 0

                    sample_hidden_states = [
                        layer_states[sample_idx][~pad_idx]
                        for layer_states in outputs.hidden_states
                    ]
                    sample_hidden_states = torch.stack(sample_hidden_states)
                    raw_sample_hidden_states.append(sample_hidden_states.squeeze().float().cpu().numpy())

        encodings = torch.cat(encoded_batches).squeeze().numpy() # shape: (num_samples, embedding_dim)
        if len(encodings.shape) == 1:
            encodings = encodings.unsqueeze(0)

        if return_raw_hidden_states:
            layerwise_encodings = torch.cat(layerwise_encoded_batches, dim=1).squeeze().numpy() # shape: (num_layers, num_samples, embedding_dim)
            return encodings, raw_sample_hidden_states, layerwise_encodings
        else:
            return encodings
    
    @torch.no_grad()
    def _get_pooled_hidden_states(self, hidden_states, attention_mask=None, method="mean"):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states[0])

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
        elif method == "first_hidden_state":
            return hidden_states[:, 0]
        else:
            raise ValueError(f"Invalid pooling method: {method}")
        
    def prepare_inputs(self, batch):
        # move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # squeeze if needed
        if len(batch['input_ids'].shape) == 3:
            batch = {k: v.squeeze() for k, v in batch.items()}

        # unsqueeze if needed, such as for augmentation dataloaders
        if len(batch['input_ids'].shape) == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        return batch
