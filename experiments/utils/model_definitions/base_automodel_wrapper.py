from typing import Any, List

import numpy as np
import torch

class BaseModelSpecifications:
    """
    Base class for model specifications. Any model specificatiosn should inherit from this class 
    and implement the additional_checks method as needed.

    Args:
        model_family (str): The huggingface model family of the model.
        model_size (str): The size of the model.
        revision (str): The huggingface revision of the model.
    """
    def __init__(self, model_family, model_size, revision, ignore_checks=False):
        self.model_family = model_family
        self.model_size = model_size
        self.revision = revision

        if not ignore_checks:
            self.do_checks()
    
    def do_checks(self):
        self.additional_checks()

    def additional_checks(self):
        raise NotImplementedError("This is a base class, please implement the additional_checks method")
    
    def __str__(self):
        return f"""
        Model family: {self.model_family}
        Model size: {self.model_size}
        Revision: {self.revision}
        """
    
class BaseLayerwiseAutoModelWrapper:
    """
    Base class for a wrapper around a huggingface model. 
    This wrapper allows for evaluating a particular layer of the model 
    with the evaluation_layer_idx parameter.

    This is a base class which should be inherited for a particular domain.

    Args:
        model_specs (BaseModelSpecifications): The model specifications.
        device_map (str): The device map to use for the model.
        evaluation_layer_idx (int): The index of the layer to use for evaluation.
    """
    def __init__(self, model_specs: BaseModelSpecifications, device_map="auto", evaluation_layer_idx: int = -1):
        model_path = model_specs.model_path_func(model_specs.model_family, model_specs.model_size)
        self.model_path = model_path
        self.model_specs = model_specs
        self.device_map = device_map
        self.evaluation_layer_idx = evaluation_layer_idx

        self.setup_model()
        self.setup_input_processor()

        #self.update_evaluation_layer(self.evaluation_layer_idx)

    """
    BASE FUNCTIONS
    """
    def update_evaluation_layer(self):
        if self.evaluation_layer_idx == -1:
            self.evaluation_layer_idx = self.num_layers - 1
        else:
            self.evaluation_layer_idx = self.evaluation_layer_idx

        assert self.evaluation_layer_idx >= 0 and self.evaluation_layer_idx < self.num_layers, \
            f"Evaluation layer {self.evaluation_layer_idx} is not in the range of the model's hidden layers 0 to {self.num_layers - 1}"
    

    def _get_hf_device_map(self):
        if hasattr(self.model, 'hf_device_map'):
            return self.model.hf_device_map
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'hf_device_map'):
            return self.model.model.hf_device_map
        elif self.model_specs.model_family == 'bert' and hasattr(self.model, 'device'):
            # BERT needs special handling because the device map is not supported
            # https://github.com/huggingface/transformers/issues/25296
            return {'device': self.model.device}
        elif hasattr(self.model, 'device'):
            return {'device': self.model.device}
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
            return {'device': self.model.model.device}
        else:
            # Use device-agnostic fallback instead of hardcoded CUDA
            import torch
            if torch.cuda.is_available():
                device = 'cuda:0'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            return {'device': device} #pray
    
    def _get_first_layer_device(self):
        device_map = self._get_hf_device_map()
        first_layer_name = list(device_map.keys())[0]
        return device_map[first_layer_name]
    
    def _model_is_nested(self):
        """
        Returns True if the model is nested, meaning that the actual model is self.model.model.
        Returns False otherwise
        """
        return hasattr(self.model, 'model')
    
    def _get_model_with_forward_pass(self):
        if self._model_is_nested():
            return self.model.model
        else:
            return self.model
    
    @property
    def device(self):
        return self._get_first_layer_device()
    
    @property
    def dtype(self):
        if self._model_is_nested():
            return self.model.model.dtype
        elif hasattr(self.model, 'dtype'):
            return self.model.dtype
        else:
            return torch.float32
    
    def forward(self, **kwargs):
        model_with_forward_pass = self._get_model_with_forward_pass()
        return model_with_forward_pass(**kwargs)
    
    def __call__(self, **kwargs):
        return self.forward(**kwargs)
    
    def print_loading_message(self):
        print(f"Loaded {self.model_path}")
        print(f"Evaluation layer: {self.evaluation_layer_idx}")
        print(f"Device: {self.device}, Number of GPUs: {torch.cuda.device_count()}")

    """
    ABSTRACT FUNCTIONS WHICH MUST BE IMPLEMENTED BY SUBCLASSES
    """
    @torch.no_grad()
    def encode(
        self,
        input_data: List,
        **kwargs: Any
    ) -> np.ndarray:
        raise NotImplementedError("This is a base class, please implement the encode method")

    def setup_input_processor(self):
        raise NotImplementedError("This is a base class, please implement the setup_input_processor method")
    
    def setup_model(self):
        raise NotImplementedError("This is a base class, please implement the setup_model method")
    
    def prepare_inputs(self, batch):
        raise NotImplementedError("This is a base class, please implement the prepare_inputs method")