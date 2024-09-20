adverse drug reactions dataset.

Commonly used dataset for RE

has page on papers with code:

https://paperswithcode.com/sota/relation-extraction-on-ade-corpus




Can load from hugging face:

from datasets import load_dataset

# Load the ADE dataset
dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification")

# Print some information about the dataset
print("Dataset structure:", dataset)
print("Sample data:", dataset['train'][0])


