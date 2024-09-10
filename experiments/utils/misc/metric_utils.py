import torch
import math
import tqdm
from .model_dataloader_utils import normalize
import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent
import numpy as np

# By default enabled, can be disabled by setting to True in the notebook
DISABLE_TQDM = False

def entropy_normalization(entropy, normalization, N, D):
    assert normalization in ['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw']

    if normalization == 'maxEntropy':
        entropy /= min(math.log(N), math.log(D))
    elif normalization == 'logN':
        entropy /= math.log(N)
    elif normalization == 'logD':
        entropy /= math.log(D)
    elif normalization == 'logNlogD':
        entropy /= (math.log(N) * math.log(D))
    elif normalization == 'raw':
        pass

    return entropy

def compute_per_forward_pass(model, dataloader, compute_function, max_samples=1000, **kwargs):
    results = {}
    counter = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=max_samples, disable=DISABLE_TQDM):
            counter += 1
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            hidden_states = [normalize(x.squeeze()) for x in outputs.hidden_states]

            batch_result = compute_function(hidden_states, **kwargs)
            for norm, values in batch_result.items():
                if norm not in results:
                    results[norm] = []
                results[norm].append(values)

            if counter >= max_samples:
                break

    return {norm: np.array(values).mean(axis=0) for norm, values in results.items()}

def compute_on_concatenated_passes(model, dataloader, compute_function, max_samples=1000, **kwargs):
    all_hidden_states = []
    counter = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=max_samples, disable=DISABLE_TQDM):
            counter += 1
            if not isinstance(batch, tuple):
                batch = (batch,)
            
            batch_hidden_states = []
            for sub_batch in batch:
                if len(batch) == 1:
                    sub_batch = {k: v.to(model.device) for k, v in sub_batch.items()}
                else:
                    sub_batch = {k: v.unsqueeze(0).to(model.device) for k, v in sub_batch.items()}
                
                outputs = model(**sub_batch)
                hidden_states = [normalize(x.squeeze()) for x in outputs.hidden_states] # L x NUM_TOKENS x D
                layer_means = torch.stack([torch.mean(x, dim=0) for x in hidden_states]) # L x D
                batch_hidden_states.append(layer_means)
            
            all_hidden_states.append(torch.stack(batch_hidden_states)) # NUM_AUG x L x D
            if counter >= max_samples:
                break

    concatenated_states = torch.stack(all_hidden_states) # NUM_SAMPLES x NUM_AUG x L x D
    concatenated_states = concatenated_states.permute(2, 0, 1, 3) # L x NUM_SAMPLES x NUM_AUG x D
    concatenated_states = concatenated_states.squeeze()
    return compute_function(concatenated_states, **kwargs)

def compute_entropy(hidden_states, alpha=1, normalizations=['maxEntropy']):
    print(hidden_states.shape)

    L, N, D = hidden_states.shape

    if N > D:
        cov = torch.matmul(hidden_states.transpose(1, 2), hidden_states) # L x N x N
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(1, 2)) # L x D x D

    cov = torch.clamp(cov, min=0)
    entropies = [itl.matrixAlphaEntropy(LAYER_COV.double(), alpha=alpha).item() for LAYER_COV in cov]

    return {norm: [entropy_normalization(x, norm, N, D) for x in entropies] for norm in normalizations}

def compute_lidar(hidden_states, alpha=1, normalizations=['maxEntropy'], return_within_scatter=False):
    L, NUM_SAMPLES, NUM_AUG, D = hidden_states.shape

    lda_matrices = [compute_LDA_matrix(layer.double(), return_within_class_scatter=return_within_scatter) for layer in hidden_states]
    entropies = [itl.matrixAlphaEntropy(lda_matrix, alpha=alpha).item() for lda_matrix in lda_matrices]
    return {norm: [entropy_normalization(x, norm, NUM_SAMPLES, D) for x in entropies] for norm in normalizations}

def compute_dime(hidden_states, alpha=1, normalizations=['maxEntropy']):
    L, N, D = hidden_states.shape
    if N > D:
        cov = torch.matmul(hidden_states.transpose(1, 2), hidden_states)
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(1, 2))
    dimes = [dent.doe(cov[0].double(), cov[1].double(), alpha=alpha, n_iters=10).item() for cov in cov]
    return {norm: [entropy_normalization(x, norm, N, D) for x in dimes] for norm in normalizations}

def compute_curvature(hidden_states, k=1):
    L, N, D = hidden_states

    def calculate_paired_curvature(a, b):
        return torch.arccos(a.T @ b).item()

    def calculate_layer_average_k_curvature(layer_p):
        summation, counter = 0, 0
        while (counter+k) < layer_p.shape[0]:
            summation += calculate_paired_curvature(layer_p[counter,:], layer_p[counter+k,:])
            counter += 1
        return summation / counter if counter > 0 else 0

    return {'raw': [calculate_layer_average_k_curvature(layer) / math.log(D) for layer in hidden_states]}

# Implements LDA matrix as defined in LIDAR paper (https://arxiv.org/pdf/2312.04000)
def compute_LDA_matrix(augmented_prompt_tensors, return_within_class_scatter=False):
    # augmented_prompt_tensors is tensor that is NUM_SAMPLES x NUM_AUGMENTATIONS x D
    NUM_SAMPLES, NUM_AUGMENTATIONS, D = augmented_prompt_tensors.shape

    delta = 1e-4

    dataset_mean = torch.mean(augmented_prompt_tensors, dim=(0, 1)).squeeze() # D
    class_means = torch.mean(augmented_prompt_tensors, dim=1) # NUM_SAMPLES x D


    # Equation 1 in LIDAR paper
    between_class_scatter = torch.zeros((D, D)).to(augmented_prompt_tensors.device)
    for i in range(NUM_SAMPLES):
        between_class_scatter += torch.outer(class_means[i] - dataset_mean, class_means[i] - dataset_mean)
    between_class_scatter /= NUM_SAMPLES

    # Equation 2 in LIDAR paper
    within_class_scatter = torch.zeros((D, D)).to(augmented_prompt_tensors.device)
    for i in range(NUM_SAMPLES):
        for j in range(NUM_AUGMENTATIONS):
            within_class_scatter += torch.outer(augmented_prompt_tensors[i, j] - class_means[i], augmented_prompt_tensors[i, j] - class_means[i])
    within_class_scatter /= (NUM_SAMPLES * NUM_AUGMENTATIONS)
    within_class_scatter += delta * torch.eye(D).to(augmented_prompt_tensors.device)

    if return_within_class_scatter:
        return within_class_scatter 
    
    # Equation 3 in LIDAR paper
    eigs, eigvecs = torch.linalg.eigh(within_class_scatter)
    within_sqrt = torch.diag(eigs**(-0.5))
    fractional_inverse = eigvecs @ within_sqrt @ eigvecs.T
    LDA_matrix = fractional_inverse @ between_class_scatter @ fractional_inverse

    return LDA_matrix