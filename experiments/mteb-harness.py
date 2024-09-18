import logging
from typing import Any, Callable, List, Literal, Type, Dict, Union
from pathlib import Path
import argparse

import numpy as np
import torch
import mteb

from utils.misc.model_dataloader_utils import get_model_path, model_name_to_sizes
from utils.model_definitions.mteb_automodel_wrapper import AutoModelWrapper, ModelSpecifications

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', type=str, default='Pythia')
    parser.add_argument('--model_size', type=str, default='1b')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--evaluation_layer', type=int, default=17, help='Layer to use for evaluation. -1 for the final layer. This is 0-indexed.')
    return parser.parse_args()

def main():
    args = parse_args()
    model_family = args.model_family
    model_size = args.model_size
    revision = args.revision
    evaluation_layer = args.evaluation_layer
    print(f"Running evaluation for {model_family} {model_size} layer {evaluation_layer}")
    
    # handle model
    model_specs = ModelSpecifications(model_family, model_size, revision=revision)
    model = AutoModelWrapper(model_specs, evaluation_layer_idx=evaluation_layer)

    # handle output folder
    results_output_folder = f'results/{model_family}/{model_size}/{revision}/mteb/layer_{model.evaluation_layer_idx}'
    def custom_create_output_folder(*args):
        output_folder = Path(results_output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder

    # handle tasks
    mteb_eng = mteb.get_benchmark("MTEB(eng)")
    reduced_mteb_eng_tasks = [task for task in mteb_eng if task.metadata.category != 'p2p']
    reduced_mteb_eng_tasks = [task for task in reduced_mteb_eng_tasks if task.metadata.type != 'Retrieval']

    # do the tasks
    encoding_kwargs = {'verbose': True}
    evaluation = mteb.MTEB(tasks=reduced_mteb_eng_tasks)
    evaluation.create_output_folder = custom_create_output_folder
    evaluation.run(model, kwargs=encoding_kwargs, output_folder='./mteb-results', raise_error=False, overwrite_results=False, verbosity=2)

if __name__ == "__main__":
    main()
