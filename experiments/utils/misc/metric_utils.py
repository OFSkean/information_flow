import torch
import math
import tqdm
from .model_dataloader_utils import normalize
import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent
import numpy as np

# By default enabled, can be disabled by setting to True in the notebook
DISABLE_TQDM = False

class EvaluationMetricSpecifications:
    def __init__(
        self, 
        evaluation_metric, 
        num_samples = 1000, 
        alpha = 1, 
        normalizations = ['maxEntropy', 'raw', 'logN', 'logNlogD', 'logD'],
        curvature_k = 1
    ):
        self.evaluation_metric = evaluation_metric
        self.num_samples = num_samples

        
        if self.evaluation_metric == 'sentence-entropy':
            self.granularity = 'sentence'
            self.evaluation_metric = 'entropy'
        elif self.evaluation_metric == 'dataset-entropy':
            self.granularity = 'dataset'
            self.evaluation_metric = 'entropy'
        else:
            self.granularity = None

        # for matrix-based metrics (LIDAR, DIME, entropy)
        self.normalizations = normalizations
        self.alpha = alpha

        # for curvature
        self.curvature_k = curvature_k
        
        self.do_checks()

    def do_checks(self):
        assert self.evaluation_metric in metric_name_to_function.keys()
        assert self.granularity in ['sentence', 'dataset', None]

        assert self.alpha > 0
        assert self.num_samples > 0
        assert self.curvature_k > 0 and isinstance(self.curvature_k, int)

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
    """
    Compute a metric for each forward pass through the model.

    Args:
        model (torch.nn.Module): The model to use for forward passes.
        dataloader (torch.utils.data.DataLoader): The dataloader providing batches.
        compute_function (callable): The function to compute the metric.
        max_samples (int): The maximum number of samples to process.
        **kwargs: Additional keyword arguments to pass to compute_function.

    Returns:
        dict: A dictionary of computed metrics, averaged over all samples.
    """
    results = {}
    counter = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=max_samples, disable=DISABLE_TQDM):
            counter += 1
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            
            hidden_states = [normalize(x.squeeze()) for x in outputs.hidden_states]
            hidden_states = torch.stack(hidden_states) # L x NUM_TOKENS x D

            batch_result = compute_function(hidden_states, **kwargs)
            for norm, values in batch_result.items():
                if norm not in results:
                    results[norm] = []
                results[norm].append(values)

            if counter >= max_samples:
                break

    return {norm: np.array(values).mean(axis=0) for norm, values in results.items()}

def compute_on_concatenated_passes(model, dataloader, compute_function, max_samples=1000, **kwargs):
    """
    Compute a metric on concatenated hidden states from multiple forward passes.

    Args:
        model (torch.nn.Module): The model to use for forward passes.
        dataloader (torch.utils.data.DataLoader): The dataloader providing batches.
        compute_function (callable): The function to compute the metric.
        max_samples (int): The maximum number of samples to process.
        **kwargs: Additional keyword arguments to pass to compute_function.

    Returns:
        dict: A dictionary of computed metrics.
    """
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
    L, N, D = hidden_states.shape

    if N > D:
        cov = torch.matmul(hidden_states.transpose(1, 2), hidden_states) # L x N x N
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(1, 2)) # L x D x D

    cov = torch.clamp(cov, min=0)
    entropies = [itl.matrixAlphaEntropy(LAYER_COV.double(), alpha=alpha).item() for LAYER_COV in cov]

    return {norm: [entropy_normalization(x, norm, N, D) for x in entropies] for norm in normalizations}

def compute_lidar(hidden_states, alpha=1, normalizations=['maxEntropy'], return_within_scatter=False):
    """
    Compute the LIDAR metric for hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states to compute LIDAR for.
        alpha (float): The alpha parameter for entropy calculation.
        normalizations (list): List of normalization methods to apply.
        return_within_scatter (bool): Whether to return the within-class scatter matrix.

    Returns:
        dict: A dictionary of computed LIDAR metrics for each normalization method.
    """
    L, NUM_SAMPLES, NUM_AUG, D = hidden_states.shape

    lda_matrices = [compute_LDA_matrix(layer.double(), return_within_class_scatter=return_within_scatter) for layer in hidden_states]
    entropies = [itl.matrixAlphaEntropy(lda_matrix, alpha=alpha).item() for lda_matrix in lda_matrices]
    return {norm: [entropy_normalization(x, norm, NUM_SAMPLES, D) for x in entropies] for norm in normalizations}

def compute_dime(hidden_states, alpha=1, normalizations=['maxEntropy']):
    """
    Compute the DIME metric for hidden states.

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
    
    dimes = [
        dent.doe(augmentation_A_covs[idx].double(), augmentation_B_covs[idx].double(), alpha=alpha, n_iters=10).item() 
        for idx in range(L)
    ]
    return {norm: [entropy_normalization(x, norm, NUM_SAMPLES, D) for x in dimes] for norm in normalizations}

def compute_infonce(hidden_states, temperature=0.1):
    """
    Compute the InfoNCE metric for hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states to compute InfoNCE for.

    Returns:
        dict: A dictionary of computed InfoNCE metrics for each normalization method.
    """

    hidden_states = hidden_states.permute(0, 2, 1, 3)
    L, NUM_AUG, NUM_SAMPLES, D = hidden_states.shape
    assert NUM_AUG == 2

    def calculate_infonce(view_a, view_b):  
        # adapted from solo-learn SIMCLR implementation
        Z = torch.cat([view_a, view_b], dim=0)
        indices = torch.arange(NUM_SAMPLES).repeat(NUM_AUG).unsqueeze(0).to(view_a.device)
        sim = torch.exp(torch.einsum("if, jf -> ij", Z, Z) / temperature)

        pos_mask = indices.t() == indices
        pos_mask.fill_diagonal_(0)
        neg_mask = indices.t() != indices

        pos = torch.sum(sim * pos_mask, 1)
        neg = torch.sum(sim * neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss.item()

    embeddings_A = [embeddings[0].double() for embeddings in hidden_states]
    embeddings_B = [embeddings[1].double() for embeddings in hidden_states]  

    infonce_scores = [
        calculate_infonce(embeddings_A[idx], embeddings_B[idx]) 
        for idx in range(L)
    ]

    return {'raw': infonce_scores}


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
        return torch.arccos(a.T @ b).item()

    def calculate_layer_average_k_curvature(layer_p):
        summation, counter = 0, 0
        for i in range(layer_p.shape[0] - k):
            summation += calculate_paired_curvature(layer_p[i].unsqueeze(1), layer_p[i+k].unsqueeze(1))
            counter += 1
        return summation / counter if counter > 0 else 0

    curvatures = [calculate_layer_average_k_curvature(layer.double()) for layer in hidden_states]
    return { 
        'raw': curvatures,
        'logD': [x / math.log(D) for x in curvatures] 
    }

def compute_LDA_matrix(augmented_prompt_tensors, return_within_class_scatter=False):
    """
    Compute the LDA matrix as defined in the LIDAR paper.

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


metric_name_to_function = {
    'entropy': compute_entropy,
    'lidar': compute_lidar,
    'dime': compute_dime,
    'infonce': compute_infonce,
    'curvature': compute_curvature
}
