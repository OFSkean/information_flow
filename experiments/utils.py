import math
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch


datasets = ['wikitext', 'ai-medical-dataset']
model_types = ["cerebras", "EleutherAI", "Medical-Llama3", "Llama3"]

cerebras_sizes = ['111M', '256M', '590M', '1.3B', '2.7B',] # '6.7B', '13B' also exist, but dont fit in 24G
EleutherAI_sizes = ['14m', '70m', '160m', '410m', '1b', '1.4b', '2.8b',]  # '6.9b', '12b' also exist, but dont fit in 24G
medical_llama3_sizes = ['8B'] # its only 8B model
llama3_sizes = ['8B'] 

def get_model_path(name, size):
    assert name in model_types

    if name == "cerebras":
        assert size in cerebras_sizes
        return f"cerebras/Cerebras-GPT-{size}"
    elif name == "EleutherAI":
        assert size in EleutherAI_sizes
        return f"EleutherAI/pythia-{size}"
    elif name == "Medical-Llama3":
        assert size in medical_llama3_sizes
        return f"ruslanmv/Medical-Llama3-8B"
    elif name == "Llama3":
        assert size in llama3_sizes
        return f"meta-llama/Meta-Llama-3-8B"
        #return "RLHFlow/ArmoRM-Llama3-8B-v0.1"  #using this finetuned version until i get baseline llama access

def get_dataloader(tokenizer, dataset_name, split='train', context_length_ratio=1, min_length=5, max_length=None, num_samples=10000, filter_text_columns=True):
    def wikitext_tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)
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

    assert dataset_name in datasets
    assert split in ['train', 'validation']
    assert context_length_ratio <= 1

    if dataset_name == 'wikitext':
        dataset = load_dataset("wikitext", 'wikitext-103-v1')[split]
    
        # filter out unneeded samples
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
        
        # tokenize the dataset
        tokenized_dataset = dataset.map(wikitext_tokenize_function, batched=True).shuffle(seed=42)
        tokenized_dataset.set_format("torch")
        
        tokenized_dataset = tokenized_dataset.filter(is_not_wikipedia_heading) # filter out headings
        
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

    # if context_length_ratio < 1, reduce all sentences to that ratio of length
    tokenized_dataset = tokenized_dataset.map(adjust_context_length, batched=False)

    # form dataloader
    dataloader = DataLoader(tokenized_dataset, shuffle=False, drop_last=True) # something is weird with batch_size=x argument here, removing it for now
    return dataloader

# from https://github.com/waltonfuture/Matrix-Entropy
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
    return R