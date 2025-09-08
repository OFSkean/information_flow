import numpy as np
import torch
from transformers import BatchFeature, AutoModel, AutoImageProcessor, AutoConfig, CLIPVisionModel, CLIPVisionConfig
from transformers.models import dinov2
from torch.utils.data import DataLoader
import timm
from torch.utils.data import Subset
from torchvision import transforms
from typing import Any, List
import tqdm
from aim.v1.torch.models import AIMForImageClassification

from .base_automodel_wrapper import BaseModelSpecifications, BaseLayerwiseAutoModelWrapper
from .jepa.JepaEncoder import load_jepa_encoder
from ..dataloaders.vision_dataloader import prepare_dataloader

model_name_to_sizes = {
    'sam': ['base'],
    'vit_augreg': ['base'],
    'vit': ['base', 'large', 'huge'],
    'dinov1': ['base'],
    'dinov2': ['small', 'base', 'large', 'giant'],
    'dinov2-register': ['small', 'base', 'large', 'giant'],
    'mae': ['base', 'large', 'huge'],
    'deit': ['base'],
    'clip': ['base', 'large'],
    'i-jepa': ['imagenet1k', 'imagenet21k'],
    'beit': ['base', 'large'],
    'aim': ['large', 'huge', '1B', '3B'],
    'aimv2': ['large', 'huge', '1B', '3B']
}
model_types = list(model_name_to_sizes.keys())

def get_model_path(name, size):
    assert name in model_types, f"Invalid model type {name}, valid types: {model_types}"
    assert size in model_name_to_sizes[name], \
        f"Invalid size {size} for model type {name}, valid sizes: {model_name_to_sizes[name]}"
    
    if name == 'vit':
        patch_size = 16 if size != 'huge' else 14
        dataset = "-in21k" if size == 'huge' else ""
        return f"google/vit-{size}-patch{patch_size}-224{dataset}"
    elif name == 'dinov1':
        return f'facebook/dino-vitb16'
    elif name == 'dinov2':
        return f'facebook/dinov2-{size}'
    elif name == 'dinov2-register':
        return f'timm/vit_{size}_patch14_reg4_dinov2.lvd142m'
    elif name == 'mae':
        return f"facebook/vit-mae-{size}"
    elif name == 'sam':
        return "facebook/sam-vit-base"
    elif name == 'vit_augreg':
        return "timm/vit_base_patch16_224.augreg_in21k"
    elif name == 'deit':
        return "facebook/deit-base-distilled-patch16-224"
    elif name == 'clip':
        if size == 'base':
            return "openai/clip-vit-base-patch16"
        elif size == 'large':
            return "openai/clip-vit-large-patch14"
    elif name == 'i-jepa':
        return ""
    elif name == 'beit':
        return f"microsoft/beit-{size}-patch16-224"
    elif name == 'aim':
        return "apple/aim-600M"
    elif name == 'aimv2':
        return f"apple/aimv2-{size}-patch14-224"

def update_config(config, model_specs):
    if model_specs.model_family == 'mae':
        config.mask_ratio = 0.

    return config

def get_model_and_config_classes(model_specs):
    if model_specs.model_family == 'clip':
        return CLIPVisionModel, CLIPVisionConfig
    elif model_specs.model_family == 'aim':
        return AIMForImageClassification, None
    else:
        return AutoModel, AutoConfig


class VisionModelSpecifications(BaseModelSpecifications):
    def __init__(self, model_family, model_size, revision):
        super().__init__(model_family, model_size, revision)
        self.model_path_func = get_model_path

    def additional_checks(self):
        pass

class VisionLayerwiseAutoModelWrapper(BaseLayerwiseAutoModelWrapper):
    def __init__(self, 
                 model_specs: VisionModelSpecifications, 
                 device_map="auto", 
                 evaluation_layer_idx: int = -1):
        super().__init__(model_specs, device_map, evaluation_layer_idx)

    """
    FUNCTIONS FOR INITIALIZATION
    """
    def setup_input_processor(self):
        if self._is_timm_model():
           data_config = timm.data.resolve_model_data_config(self.model)
           self.image_processor = timm.data.create_transform(**data_config, is_training=False)
        elif self.model_specs.model_family in ['i-jepa', 'aim']:
            self.image_processor = lambda x: x
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def process_inputs(self, inputs):
        if self._is_timm_model():
            return self.image_processor(inputs)
        elif self.model_specs.model_family == 'i-jepa':
            return inputs
        else:
            return self.image_processor(inputs, return_tensors="pt")

    def setup_model(self):
        if self._is_timm_model():
            self.setup_timm_model()
        elif self.model_specs.model_family == 'i-jepa':
            self.setup_jepa_model()
        elif self.model_specs.model_family == 'aim':
            self.setup_aim_model()
        else:
            self.setup_huggingface_model()

    def setup_jepa_model(self):
        self.model = load_jepa_encoder(self.model_specs.model_size)

    def setup_aim_model(self):
        # with the standard AIMv1 git pull, there may be a bug in the config
        # where the img_size variable is called image_size. Manual fix for now
        self.model = AIMForImageClassification.from_pretrained(self.model_path)
        self.model.forward = lambda x: {'hidden_states': self.model.extract_features(x)}
        self.model.trunk.post_transformer_layer = None
        self.model.eval().cuda()

    def setup_huggingface_model(self):
        MODEL_CLASS, CONFIG_CLASS = get_model_and_config_classes(self.model_specs)
        if CONFIG_CLASS is None:
            self.config = None
        else:
            self.config = CONFIG_CLASS.from_pretrained(self.model_path, 
                                            revision=self.model_specs.revision,
                                            output_hidden_states=True,
                                            trust_remote_code=True)
            self.config = update_config(self.config, self.model_specs)
        
        self.num_layers = self.config.num_hidden_layers + 1 
        self.update_evaluation_layer()
        self.config.num_hidden_layers = self.evaluation_layer_idx

        FROM_PRETRAINED_KWARGS = {
            'revision': self.model_specs.revision,
            'config': self.config,
            'torch_dtype': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            'device_map': self.device_map if False else None,
            'trust_remote_code': True
        }

        self.model = MODEL_CLASS.from_pretrained(self.model_path, **FROM_PRETRAINED_KWARGS).eval()

        if FROM_PRETRAINED_KWARGS['device_map'] is None:
            self.model.to("cuda")

    def setup_timm_model(self):
        base_model_path = self.model_path.split('/')[1]
        self.model = timm.create_model(base_model_path, pretrained=True, num_classes=0)
        self.model = self.model.eval().cuda()

    def __call__(self, **kwargs):
        if 'timm' in self.model_path:
            return {
                'hidden_states': self.model.forward_intermediates(**kwargs, intermediates_only=True, output_fmt='NLC')
            }
        else:
            return self.forward(**kwargs)
        
    def prepare_inputs(self, batch, return_labels=False):
        batch_idx, images, labels = batch
            
        if isinstance(images, BatchFeature):
            inputs = images.to("cuda")
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(self.dtype)
        elif self._is_timm_model() or self.model_specs.model_family in ['i-jepa', 'aim']:
            inputs = {
                "x": images.to("cuda").to(self.dtype)
            }
        else:
            inputs = {
                "pixel_values": images.to("cuda").to(self.dtype)
            }
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        if return_labels:
            return inputs, labels
        else:
            return inputs
    
    def _is_timm_model(self):
        return 'timm' in self.model_path
    
    
    """
    FUNCTIONS FOR INFERENCE
    """
    @torch.no_grad()
    def encode(
        self,
        input_dataset: Subset,
        return_raw_hidden_states: bool = False,
        **kwargs: Any
    ) -> np.ndarray:
        verbose = kwargs.pop("verbose", True)

        dataloader = prepare_dataloader(input_dataset, 
                                        batch_size=256, 
                                        num_workers=32,
                                        shuffle=False,
                                        is_multiview=False,
                                        drop_last=False)

        if return_raw_hidden_states:
            embeddings, raw_hidden_states, layerwise_encodings, labels = self._encode_helper(dataloader, 
                                                            verbose=verbose, 
                                                            return_raw_hidden_states=return_raw_hidden_states)
            return np.array(embeddings), raw_hidden_states, layerwise_encodings, labels
        
        else:
            embeddings = self._encode_helper(dataloader, 
                                            verbose=verbose, 
                                            return_raw_hidden_states=return_raw_hidden_states) # shape: (num_samples, embedding_dim)
            return np.array(embeddings)
        

    @torch.no_grad()
    def _encode_helper(self, dataloader, verbose=False, return_raw_hidden_states=False) -> np.ndarray:
        encoded_batches = []
        layerwise_encoded_batches = []
        labels = []

        if return_raw_hidden_states:
            # can be memory intensive, so only do if needed
            raw_sample_hidden_states = []

        for batch in tqdm.tqdm(dataloader, total=len(dataloader), disable= not verbose):
            _, _, label = batch
            labels.extend(label)
            batch = self.prepare_inputs(batch)
            
            outputs = self.forward(**batch)

            hidden_states = outputs.hidden_states[self.evaluation_layer_idx]
            hidden_states = self._get_pooled_hidden_states(hidden_states, method="mean")
            encoded_batches.append(hidden_states.half().cpu())

            if return_raw_hidden_states:
                # get layerwise encodings for the batch
                current_batch_layerwise_encodings = []
                for layer_idx in range(len(outputs.hidden_states)):
                    layer_states = outputs.hidden_states[layer_idx]
                    layer_states = self._get_pooled_hidden_states(layer_states, method="mean")
                    current_batch_layerwise_encodings.append(layer_states.half().cpu())
                layerwise_encoded_batches.append(torch.stack(current_batch_layerwise_encodings))
     
                # get raw hidden states for each sample
                for sample_idx in range(len(outputs.hidden_states[0])):
                    sample_hidden_states = [
                        layer_states[sample_idx] for layer_states in outputs.hidden_states
                    ]
                    sample_hidden_states = torch.stack(sample_hidden_states)
                    raw_sample_hidden_states.append(sample_hidden_states.squeeze().half().cpu().numpy())

        encodings = torch.cat(encoded_batches).squeeze().numpy() # shape: (num_samples, embedding_dim)
        layerwise_encodings = torch.cat(layerwise_encoded_batches, dim=1).squeeze().numpy() # shape: (num_layers, num_samples, embedding_dim)

        if return_raw_hidden_states:
            return encodings, raw_sample_hidden_states, layerwise_encodings, labels
        else:
            return encodings
    
    @torch.no_grad()
    def _get_pooled_hidden_states(self, hidden_states, attention_mask, method="mean"):
        if method == "mean":
            layer_means = torch.stack([torch.mean(x, dim=0) for x in hidden_states])
            return layer_means
        
        elif method == "last_hidden_state":
            return hidden_states[:, -1]
        else:
            raise ValueError(f"Invalid pooling method: {method}")
        
    @property
    def num_features(self):
        if hasattr(self.model, "num_features"):
            return self.model.num_features
        elif hasattr(self.model, "inplanes"):
            return self.model.inplanes
        elif hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        elif type(self.model) == AIMForImageClassification:
            # for v1 600M
            return 1536
        else:
            raise ValueError("Could not find num_features or inplanes")

    @property
    def parameters(self):
        return self.model.parameters
