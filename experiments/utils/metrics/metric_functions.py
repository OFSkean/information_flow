import numpy as np
import torch
import math
import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent
from dadapy.data import Data as ID_DATA
    
def entropy_normalization(entropy, normalization, N, D):
    """
    Normalize the entropy based on the specified normalization method.

    Args:
        entropy (float): The entropy value to be normalized.
        normalization (str): The normalization method to use.
        N (int): The number of samples.
        D (int): The dimensionality of the data.

    Returns:
        float: The normalized entropy value.
    """
    assert normalization in ['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length']

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
    elif normalization == 'length':
        entropy = N

    return entropy

def hacky_collation(batch):
    ips = [item['input_ids'] for item in batch]
    attn = [item['attention_mask'] for item in batch]

    return {'input_ids': torch.stack(ips),
            'attention_mask': torch.stack(attn),
            }

# from https://github.com/waltonfuture/Matrix-Entropy
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
    return R


def compute_dime(hidden_states, alpha=1, normalizations=['maxEntropy']):
    """
    Compute the DIME metric for hidden states.
        https://arxiv.org/abs/2301.08164

    Args:
        hidden_states (torch.Tensor): The hidden states to compute DIME for.
        alpha (float): The alpha parameter for entropy calculation.
        normalizations (list): List of normalization methods to apply.

    Returns:
        dict: A dictionary of computed DIME metrics for each normalization method.
    """
    hidden_states = hidden_states.permute(0, 2, 1, 3)
    L, NUM_AUG, NUM_SAMPLES, D = hidden_states.shape
    assert NUM_AUG == 2
    
    if NUM_SAMPLES > D:
        cov = torch.matmul(hidden_states.transpose(-1, -2), hidden_states) 
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(-1, -2))

    augmentation_A_covs = [cov[0].double() for cov in cov]
    augmentation_B_covs = [cov[1].double() for cov in cov]
    
    dimes = []
    for idx in range(L):
        try:
            dimes.append(dent.doe(augmentation_A_covs[idx].double(), augmentation_B_covs[idx].double(), alpha=alpha, n_iters=10).item())
        except Exception as e:
            dimes.append(np.nan)

    return {norm: [entropy_normalization(x, norm, NUM_SAMPLES, D) for x in dimes] for norm in normalizations}

def compute_infonce(hidden_states, temperature=0.1):
    """
    Compute the InfoNCE metric for hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states to compute InfoNCE for.
        temperature (float): The temperature parameter for InfoNCE calculation.

    Returns:
        dict: A dictionary of computed InfoNCE metrics for each normalization method.
    """

    hidden_states = hidden_states.permute(0, 2, 1, 3)
    L, NUM_AUG, NUM_SAMPLES, D = hidden_states.shape
    assert NUM_AUG == 2

    def calculate_infonce(view_a, view_b):
        logits = view_a @ view_b.T

        labels = torch.arange(len(view_a), device=view_a.device)

        return torch.nn.functional.cross_entropy(logits / temperature, labels, reduction='mean').item()

    embeddings_A = [embeddings[0].double() for embeddings in hidden_states]
    embeddings_B = [embeddings[1].double() for embeddings in hidden_states]  

    infonce_scores = [
        calculate_infonce(embeddings_A[idx], embeddings_B[idx]) 
        for idx in range(L)
    ]

    normalized_infonce_scores = [1 - (x / math.log(NUM_SAMPLES)) for x in infonce_scores]

    return {'raw': infonce_scores, 'mi-lower-bound': normalized_infonce_scores}


def compute_curvature(hidden_states, k=1):
    """
    Compute the average k-step curvature of hidden states across layers.

    Args:
        hidden_states (torch.Tensor): List of hidden states tensors, one for each layer.
        k (int): The step size for curvature calculation.

    Returns:
        dict: A dictionary containing the computed average k-step curvature values for each layer.
    """
    L, N, D = hidden_states.shape

    def calculate_paired_curvature(a, b):
        dotproduct = torch.abs(a.T @ b)
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0

        argument = torch.clamp(dotproduct / (norm_a * norm_b), min=-1, max=1)
        curvature = torch.arccos(argument)
        
        if torch.isnan(curvature):
            print(a)
            print(b)
            print(curvature)
            print(dotproduct)
            print(norm_a)
            print(norm_b)
            print(dotproduct / (norm_a * norm_b))
            raise Exception("Curvature is NaN")
        return curvature.item()

    def calculate_layer_average_k_curvature(layer_p):
        summation, counter = 0, 0
        for k in range(1, layer_p.shape[0]-1):
            v_k = layer_p[k].unsqueeze(1) - layer_p[k-1].unsqueeze(1)
            v_kplusone = layer_p[k+1].unsqueeze(1) - layer_p[k].unsqueeze(1)
            curvature = calculate_paired_curvature(v_kplusone, v_k)
            summation += curvature
            counter += 1
        return summation / counter if counter > 0 else 0

    curvatures = [calculate_layer_average_k_curvature(layer.double()) for layer in hidden_states]
    return { 
        'raw': curvatures,
        'logD': [x / math.log(D) for x in curvatures] 
    }



def compute_entropy(hidden_states, alpha=1, normalizations=['maxEntropy']):
    """
    Compute the prompt entropy of the hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states to compute entropy for.
        alpha (float): The alpha parameter for entropy calculation.
        normalizations (list): List of normalization methods to apply.

    Returns:
        dict: A dictionary of computed entropy metrics for each normalization method.
    """
    L, N, D = hidden_states.shape

    if N > D:
        cov = torch.matmul(hidden_states.transpose(1, 2), hidden_states) # L x N x N
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(1, 2)) # L x D x D

    cov = torch.clamp(cov, min=0)
    entropies = []
    for layer_cov in cov:
        try:
            layer_cov = layer_cov.cpu()
            layer_cov = layer_cov.double() / torch.trace(layer_cov.double())
            entropies.append(itl.matrixAlphaEntropy(layer_cov, alpha=alpha).item())
        except Exception as e:
            entropies.append(np.nan)

    return {norm: [entropy_normalization(x, norm, N, D) for x in entropies] for norm in normalizations}


def compute_LDA_matrix(augmented_prompt_tensors, return_within_class_scatter=False):
    """
    Compute the LDA matrix as defined in the LIDAR paper.
        https://arxiv.org/abs/2312.04000

    Args:
        augmented_prompt_tensors (torch.Tensor): Tensor of augmented prompts.
        return_within_class_scatter (bool): Whether to return the within-class scatter matrix.

    Returns:
        torch.Tensor: The computed LDA matrix or within-class scatter matrix.
    """
    # augmented_prompt_tensors is tensor that is NUM_SAMPLES x NUM_AUGMENTATIONS x D
    NUM_SAMPLES, NUM_AUGMENTATIONS, D = augmented_prompt_tensors.shape

    delta = 1e-4

    dataset_mean = torch.mean(augmented_prompt_tensors, dim=(0, 1)).squeeze() # D
    class_means = torch.mean(augmented_prompt_tensors, dim=1) # NUM_SAMPLES x D

    # Equation 1 in LIDAR paper
    between_class_scatter = torch.zeros((D, D)).to(augmented_prompt_tensors.device)
    for i in range(NUM_SAMPLES):
        between_class_scatter += torch.outer(class_means[i] - dataset_mean, class_means[i] - dataset_mean)
    between_class_scatter /= NUM_SAMPLES # D x D

    # Equation 2 in LIDAR paper
    within_class_scatter = torch.zeros((D, D)).to(augmented_prompt_tensors.device)
    for i in range(NUM_SAMPLES):
        for j in range(NUM_AUGMENTATIONS):
            within_class_scatter += torch.outer(augmented_prompt_tensors[i, j] - class_means[i], augmented_prompt_tensors[i, j] - class_means[i])
    within_class_scatter /= (NUM_SAMPLES * NUM_AUGMENTATIONS) # D x D
    within_class_scatter += delta * torch.eye(D).to(augmented_prompt_tensors.device) # D x D

    if return_within_class_scatter:
        return within_class_scatter 
    
    # Equation 3 in LIDAR paper
    eigs, eigvecs = torch.linalg.eigh(within_class_scatter)
    within_sqrt = torch.diag(eigs**(-0.5))
    fractional_inverse = eigvecs @ within_sqrt @ eigvecs.T
    LDA_matrix = fractional_inverse @ between_class_scatter @ fractional_inverse # D x D

    return LDA_matrix

def compute_lidar(hidden_states, alpha=1, normalizations=['maxEntropy'], return_within_scatter=False):
    """
    Compute the LIDAR metric for hidden states.
        https://arxiv.org/abs/2312.04000

    Args:
        hidden_states (torch.Tensor): The hidden states to compute LIDAR for.
        alpha (float): The alpha parameter for entropy calculation.
        normalizations (list): List of normalization methods to apply.
        return_within_scatter (bool): Whether to return the within-class scatter matrix.

    Returns:
        dict: A dictionary of computed LIDAR metrics for each normalization method.
    """
    L, NUM_SAMPLES, NUM_AUG, D = hidden_states.shape

    lidars = []
    for layer in hidden_states:
        try:
            lda_matrix = compute_LDA_matrix(layer.double(), return_within_class_scatter=return_within_scatter)
            lidars.append(itl.matrixAlphaEntropy(lda_matrix, alpha=alpha).item())
        except Exception as e:
            lidars.append(np.nan)
    return {norm: [entropy_normalization(x, norm, NUM_SAMPLES, D) for x in lidars] for norm in normalizations}



"""
Implementation of intrinsic dimension metric using the TwoNN estimor
"""
def compute_intrinsic_dimension(hidden_states, nn=2):
    # uses the TwoNN method to estimate intrinsic dimension    
    # iterate over layers, skip the first layer

    intrinsic_dimensions = [0]
    normalized_intrinsic_dimensions = [0]
    for layer_num, layer in enumerate(hidden_states[1:]):
        layer = layer.detach().float().squeeze().cpu().numpy()
        data = ID_DATA(layer)
        id, id_error, id_distance = data.compute_id_2NN()

        intrinsic_dimensions.append(id)
        normalized_intrinsic_dimensions.append(id / math.log(layer.shape[0]))
    return {'raw': intrinsic_dimensions, 'logN': normalized_intrinsic_dimensions}