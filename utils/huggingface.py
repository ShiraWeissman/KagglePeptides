from transformers import AutoTokenizer
from datasets import load_dataset
from utils.general import *


def load_pretrained_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(saved_models_path,  tokenizer_name))
    return tokenizer


def save_trained_tokenizer(tokenizer, tokenizer_name):
    tokenizer.save_pretrained(os.path.join(saved_models_path, tokenizer_name))

def save_trained_model(model, model_name):
    model.save_pretrained(os.path.join(saved_models_path, model_name))

def tokenize_dataset(data, tokenizer):
    def tokenize_function(data):
        return tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)
    data = data.map(tokenize_function, batched=True)
    return data

if __name__ == '__main__':
    pass