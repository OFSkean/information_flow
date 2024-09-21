import logging
from typing import Any, Callable, List, Literal, Type, Dict, Union
from pathlib import Path
import argparse
import os
import pickle
from itertools import product

import numpy as np
import torch
import mteb
from transformers import AutoModel, AutoTokenizer

from utils.model_definitions.mteb_automodel_wrapper import AutoModelWrapper, ModelSpecifications
from utils.misc.metric_utils import (
    compute_per_forward_pass,
    compute_on_concatenated_passes,
    metric_name_to_function,
    EvaluationMetricSpecifications
)
from utils.misc.model_dataloader_utils import (
    model_name_to_sizes, 
    get_model_path, 
    get_dataloader, 
    get_augmentation_collated_dataloader
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', type=str, default='Pythia')
    parser.add_argument('--model_size', type=str, default='14m')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--evaluation_layer', type=int, default=-1, help='Layer to use for evaluation. -1 for the final layer. This is 0-indexed.')
    parser.add_argument('--base_results_path', type=str, default='experiments/results')
    parser.add_argument('--purpose', type=str, default='run_tasks', choices=['run_tasks', 'run_entropy_metrics', 'download_datasets'])
    parser.add_argument('--raise_error', type=bool, default=False)
    return parser.parse_args()


def get_results_path(model_specs: ModelSpecifications, evaluation_metric_specs: EvaluationMetricSpecifications, dataloader_kwargs, base_results_path, include_file_name=True):
    model_family = model_specs.model_family
    model_size = model_specs.model_size
    revision = model_specs.revision
    evaluation_metric = evaluation_metric_specs.evaluation_metric
    granularity = evaluation_metric_specs.granularity
    dataset = dataloader_kwargs['dataset_name']
    split = dataloader_kwargs['split']

    if evaluation_metric == 'entropy':
        evaluation_metric = f"{evaluation_metric}_{granularity}"

    if include_file_name:
        return f"{base_results_path}/{model_family}/{model_size}/{revision}/metrics/{dataset}/{split}/{evaluation_metric}.pkl"
    else:
        return f"{base_results_path}/{model_family}/{model_size}/{revision}/metrics/{dataset}/{split}"

def save_results(results, model_specs: ModelSpecifications, evaluation_metric_specs: EvaluationMetricSpecifications, dataloader_kwargs, base_results_path):
    results_path = get_results_path(model_specs, evaluation_metric_specs, dataloader_kwargs, base_results_path, include_file_name=False)
    evaluation_metric = evaluation_metric_specs.evaluation_metric
    if evaluation_metric == 'entropy':
        evaluation_metric = f"{evaluation_metric}_{evaluation_metric_specs.granularity}"

    os.makedirs(results_path, exist_ok=True)
    with open(f"{results_path}/{evaluation_metric}.pkl", "wb") as f:
        pickle.dump(results, f)

def calculate_and_save_layerwise_metrics(
    model,
    model_specs: ModelSpecifications,
    evaluation_metric_specs: EvaluationMetricSpecifications,
    dataloader_kwargs: Dict[str, Any],
    base_results_path: str
):
    tokenizer = model.tokenizer

    if evaluation_metric_specs.evaluation_metric == 'entropy':
        dataloader = get_dataloader(tokenizer, **dataloader_kwargs)
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations
        }
        forward_pass_func = compute_per_forward_pass if evaluation_metric_specs.granularity == 'sentence' else compute_on_concatenated_passes
  

    elif evaluation_metric_specs.evaluation_metric == 'curvature':
        dataloader = get_dataloader(tokenizer, **dataloader_kwargs)
        compute_func_kwargs = {
            'k': evaluation_metric_specs.curvature_k
        }
        forward_pass_func = compute_per_forward_pass

    elif evaluation_metric_specs.evaluation_metric == 'lidar':
        dataloader_kwargs['num_augmentations_per_sample'] = 16
        dataloader = get_augmentation_collated_dataloader(tokenizer, **dataloader_kwargs)
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations,
        }
        forward_pass_func = compute_on_concatenated_passes

    elif evaluation_metric_specs.evaluation_metric == 'dime':
        dataloader_kwargs['num_augmentations_per_sample'] = 2
        dataloader = get_augmentation_collated_dataloader(tokenizer, **dataloader_kwargs)
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations,
        }
        forward_pass_func = compute_on_concatenated_passes

    elif evaluation_metric_specs.evaluation_metric == 'infonce':
        dataloader_kwargs['num_augmentations_per_sample'] = 2
        dataloader = get_augmentation_collated_dataloader(tokenizer, **dataloader_kwargs)
        compute_func_kwargs = {
            'temperature': 0.1,
        }
        forward_pass_func = compute_on_concatenated_passes

    compute_func = metric_name_to_function[evaluation_metric_specs.evaluation_metric]
    results = forward_pass_func(model, dataloader, compute_func, **compute_func_kwargs)
    save_results(results, model_specs, evaluation_metric_specs, dataloader_kwargs, base_results_path)


def run_entropy_metrics(model, model_specs: ModelSpecifications, MTEB_evaluator: mteb.MTEB, args):
    task_datasets = [task.metadata.dataset['path'] for task in MTEB_evaluator.tasks]
    metrics = ['infonce', 'dime', 'lidar', 'sentence-entropy', 'dataset-entropy']
    splits = ['train', 'test']
    for task_dataset, metric, split in product(task_datasets, metrics, splits):
        print(f"Running evaluation for {task_dataset} - {metric} - {split}")
        evaluation_metric_specs = EvaluationMetricSpecifications(evaluation_metric=metric)

        dataloader_kwargs = {
            'dataset_name': task_dataset,
            'split': split,
            'num_samples': 10000
        }

        # Check if results already exist
        results_path = get_results_path(model_specs, evaluation_metric_specs, dataloader_kwargs, args.base_results_path, include_file_name=True)
        if os.path.exists(results_path):
            print(f"Results already exist for {task_dataset} - {metric} - {split}. Skipping...")
            continue
        
        try:
            calculate_and_save_layerwise_metrics(model, model_specs, evaluation_metric_specs, dataloader_kwargs, args.base_results_path)
        except Exception as e:
            print(f"Error running evaluation for {task_dataset} - {metric} - {split}: {str(e)}")

def main():
    args = parse_args()
    model_family = args.model_family
    model_size = args.model_size
    revision = args.revision
    evaluation_layer = args.evaluation_layer

    print(f"Running evaluation for {model_family} {model_size} {revision} layer {evaluation_layer}")
    model_specs = ModelSpecifications(model_family, model_size, revision=revision)

    # handle tasks
    mteb_eng = mteb.get_benchmark("MTEB(eng)")
    reduced_mteb_eng_tasks = [task for task in mteb_eng if task.metadata.category != 'p2p']
    reduced_mteb_eng_tasks = [task for task in reduced_mteb_eng_tasks if task.metadata.type != 'Retrieval']
    evaluator = mteb.MTEB(tasks=reduced_mteb_eng_tasks)
    
    device_map = "auto" if model_family != 'bert' else None
    model = AutoModelWrapper(model_specs, device_map=device_map, evaluation_layer_idx=evaluation_layer)

    if args.purpose == 'run_tasks': 
        results_output_folder = f'{args.base_results_path}/{model_family}/{model_size}/{revision}/mteb/layer_{model.evaluation_layer_idx}'
        def custom_create_output_folder(*args):
            output_folder = Path(results_output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            return output_folder
        
        encoding_kwargs = {'verbose': True}
        evaluator.create_output_folder = custom_create_output_folder
        evaluator.run(model, kwargs=encoding_kwargs, output_folder='./mteb-results', raise_error=args.raise_error, overwrite_results=False, verbosity=2)

    elif args.purpose == 'run_entropy_metrics':
        run_entropy_metrics(model, model_specs, evaluator, args)

    elif args.purpose == 'download_datasets':
        for task in evaluator.tasks:
            task.load_data()


if __name__ == "__main__":
    main()
