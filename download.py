# This file runs during container build time to get model weights built into the container

# set sys path to include the parent directory

# In this example: A Huggingface BERT model
from transformers import AutoModelForCausalLM , AutoTokenizer

def download_model():
    tokenizer = AutoTokenizer.from_pretrained("sahil2801/test3", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("sahil2801/test3", trust_remote_code=True)
    # do a dry run of loading the huggingface model, which will download weights

if __name__ == "__main__":
    download_model()