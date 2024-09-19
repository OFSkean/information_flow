import os
import math
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap
import tqdm
import warnings
import mteb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

datasets = ['wikitext', 'ai-medical-dataset']
model_types = ["cerebras", "Pythia", "mamba", "mamba2", "Medical-Llama3", "Llama3", "bert"]

cerebras_sizes = ['111M', '256M', '590M', '1.3B', '2.7B', '6.7B', '13B'] # '13b' also exists but doesnt fit in 24G for bfloat16
Pythia_sizes = ['14m', '70m', '160m', '410m', '1b', '1.4b', '2.8b']#, '6.9b'] # '12b' also exists but doesnt fit in 24G for bfloat16
mamba_sizes = ['130m', '370m', '790m', '1.4b', '2.8b']
mamba2_sizes = ['130m', '370m', '780m', '1.3b', '2.7b']
bert_sizes = ['base', 'large']
medical_llama3_sizes = ['8B'] # its only 8B model
llama3_sizes = ['8B'] 

model_name_to_sizes = {
    'Pythia': Pythia_sizes,
    'cerebras': cerebras_sizes,
    'mamba': mamba_sizes,
    'mamba2': mamba2_sizes,
    'Medical-Llama3': medical_llama3_sizes,
    'Llama3': llama3_sizes,
    'bert': bert_sizes
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

def get_dataloader(
        tokenizer, 
        dataset_name, 
        split='train', 
        context_length_ratio=1, 
        min_length=5,
        max_length=None, 
        num_samples=10000, 
        filter_text_columns=True, 
        augment=False,
        return_dataset=False,
        max_sample_length=2048,
        num_workers=8
    ):
    
    def general_tokenize_function(examples):
        if not augment:
            texts = examples["text"]
        else:
            texts = text_augmentation(examples["text"]) 

        return tokenizer(texts, truncation=True, max_length=max_sample_length)
    
    def medical_tokenize_function(examples):
        medical_prompt = """You are an AI Medical Assistant Chatbot, trained to answer medical questions. Below is an instruction that describes a task, paired with an response context. Write a response that appropriately completes the request.

            ### Instruction:
            {}


            ### Response:
            {}"""
        
        instructions = examples["question"]
        outputs      = examples["context"]
        texts = []
        for instruction, output in zip(instructions,  outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = medical_prompt.format(instruction,  output)
            texts.append(text)

        return tokenizer(texts, truncation=True, max_length=1000)
    
    def adjust_context_length(examples):
        if context_length_ratio == 1:
            return examples
        else:
            input_length = len(examples['input_ids'])
            context_length = max(2, int(input_length * context_length_ratio))
            examples['attention_mask'] = examples['attention_mask'][:context_length]
            examples['input_ids'] = examples['input_ids'][:context_length]

            return examples

    def is_not_wikipedia_heading(example):
        return not (example["text"].strip().startswith("=") and example["text"].strip().endswith("="))

    assert dataset_name in datasets or 'mteb' in dataset_name
    assert context_length_ratio <= 1

    if dataset_name == 'wikitext':
        dataset = load_dataset("wikitext", 'wikitext-103-v1')[split]
    
        # filter out unneeded samples
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
        dataset = dataset.filter(is_not_wikipedia_heading) # filter out headings
        
        # filter out samples by lower bound and upper bound on length
        dataset = dataset.filter(lambda x: len(x['text']) >= 2*min_length) # filter out the frequent blank/small examples in the dataset
        if max_length is not None:
            dataset = dataset.filter(lambda x: len(x['text']) <= 2*max_length)

        # tokenize the dataset
        try:
            tokenized_dataset = dataset.map(general_tokenize_function, batched=True).shuffle(seed=42)
            tokenized_dataset.set_format("torch")
        except Exception as e:
            for idx, d in enumerate(dataset):
                print(idx, d)
            raise e
        
        if filter_text_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    elif dataset_name == 'ai-medical-dataset':
        dataset = load_dataset("ruslanmv/ai-medical-dataset")[split]
    
        # filter out unneeded samples
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

        # tokenize the dataset
        tokenized_dataset = dataset.map(medical_tokenize_function, batched=True).shuffle(seed=42)
        tokenized_dataset.set_format("torch")

        if filter_text_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(["question"])
            tokenized_dataset = tokenized_dataset.remove_columns(["context"])

        # filter out samples by lower bound and upper bound on length
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) >= min_length) # filter out the frequent blank/small examples in the dataset
        if max_length is not None:
            tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) <= max_length)

    elif 'mteb' in dataset_name:
        dataset = load_dataset(dataset_name)[split]

        # filter out unneeded samples
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

        tokenized_dataset = dataset.map(general_tokenize_function, batched=True).shuffle(seed=42)
        tokenized_dataset.set_format("torch")

        if filter_text_columns:
            for column in tokenized_dataset.column_names:
                if column not in ['input_ids', 'attention_mask']:
                    tokenized_dataset = tokenized_dataset.remove_columns([column])


    # if context_length_ratio < 1, reduce all sentences to that ratio of length
    tokenized_dataset = tokenized_dataset.map(adjust_context_length, batched=False)

    if return_dataset:
        return tokenized_dataset
    
    # form dataloader
    dataloader = DataLoader(tokenized_dataset, shuffle=False, num_workers=16) # something is weird with batch_size=x argument here, removing it for now
    return dataloader

def get_augmentation_collated_dataloader(
        tokenizer, 
        dataset_name, 
        split='train',
        num_augmentations_per_sample=8,
        context_length_ratio=1, 
        min_length=2,
        max_length=None, 
        num_samples=10000, 
        filter_text_columns=True,
        max_sample_length=2048,
        num_workers=8
    ):

    base_datasets = [
        get_dataloader(
            tokenizer, 
            dataset_name, 
            split=split, 
            context_length_ratio=context_length_ratio, 
            min_length=min_length,
            max_length=max_length, 
            num_samples=num_samples, 
            filter_text_columns=filter_text_columns, 
            augment=True,
            return_dataset=True,
            max_sample_length=max_sample_length,
            num_workers=num_workers
        ) for _ in range(num_augmentations_per_sample)
    ]

    lengths = [len(d) for d in base_datasets]
    assert all([l == lengths[0] for l in lengths])

    dataset_iterator = zip(*base_datasets)

    return dataset_iterator

# from https://github.com/waltonfuture/Matrix-Entropy
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
    return R


def embed_sentences_and_get_outputs(model, tokenizer, sentences: list[str]):
    tokenized_string= tokenizer(sentences, truncation=False, return_tensors='pt')
    tokenized_string = {k: v.to(model.device) for k, v in tokenized_string.items()}
    with torch.no_grad():
        outputs = model(**tokenized_string)
    
    outputs['input_ids'] = list(tokenized_string['input_ids'])
    return outputs


def reduce_and_visualize_hidden_states(hidden_states, reduction="tsne", labels=None):
    assert reduction in ["tsne", "umap"]
    warnings.filterwarnings(action='ignore', category=UserWarning)

    layers_per_row = 5
    column_width, row_height = 3, 3

    num_layers = len(hidden_states)
    num_rows = math.ceil(num_layers / layers_per_row)
    fig, axs = plt.subplots(num_rows, layers_per_row, figsize=(row_height*layers_per_row, column_width*num_rows))
    num_tokens = hidden_states[0].shape[1]

    print("NUM LAYERS", num_layers)

    # reduce and plot hidden states at each layer
    # go in reverse to make sure that dimensionality reduction has good initialization
    reduced_embeddings_by_layer = []
    for i in tqdm.tqdm(list(reversed(range(num_layers)))):
        row, col = divmod(i, layers_per_row)

        layer_hidden_states = hidden_states[i].squeeze().cpu().numpy()

        if reduction == "tsne":
            if len(reduced_embeddings_by_layer):
                # for some consistency between layers
                tsne_reducer = TSNE(n_components=2, perplexity=20, random_state=0, metric="cosine", init=reduced_embeddings_by_layer[-1])
            else:
                tsne_reducer = TSNE(n_components=2, perplexity=20, random_state=0, metric="cosine", init="pca")
            reduced_results = tsne_reducer.fit_transform(layer_hidden_states)
            reduced_embeddings_by_layer.append(reduced_results)
        elif reduction == "umap":
            if len(reduced_embeddings_by_layer):
                # for some consistency between layers
                umap_reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=0, init=reduced_embeddings_by_layer[-1])
            else:
                umap_reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=0, init="spectral")
            reduced_results = umap_reducer.fit_transform(layer_hidden_states)
            reduced_embeddings_by_layer.append(reduced_results)

        if labels is None:          
            colors = np.array(list(range(num_tokens)))
        else:
            colors = labels
        
        # plot reduced embeddings
        axs[row][col].scatter(reduced_results[:, 0], reduced_results[:, 1], c=colors, cmap="viridis")
        axs[row][col].text(0.95, 0.95, f"Layer {i}", transform=axs[row][col].transAxes, ha="left", va="top") # put row number in corner
        axs[row][col].axis("off")   # hide axes


    # hide empty plots
    for i in range(num_layers, num_rows*layers_per_row):
        row, col = divmod(i, layers_per_row)
        axs[row][col].axis("off")

    fig.show()

    # unreverse the reduced embeddings
    reduced_embeddings_by_layer = list(reversed(reduced_embeddings_by_layer))
    return reduced_embeddings_by_layer

def text_augmentation(texts, num_augmentations_per_sample=1):
    # input is list of strings
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    import nlpaug.flow as naf

    aug = naf.Sequential([
        naw.SplitAug(),
        nac.RandomCharAug(),
        nac.KeyboardAug()
    ])

    augmented_text = [str(aug.augment(x, n=num_augmentations_per_sample)) for x in texts]

    return augmented_text
