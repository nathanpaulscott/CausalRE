source URL: https://cogcomp.seas.upenn.edu/page/resource_view/43

col0 has the sentence id, each row is a word, the word is in col5
col1 is the entity class label
col2 is the word pos in the sentence
col4 is the pos tag


Many sentences have no relations when thay actually do

The only relations they are mapping are:
located in
work for
organization based in
live in
kill

So this is a pretty limited dataset!!!!




can just get from HF:


from datasets import load_dataset

# Load the CoNLL-2004 dataset
dataset = load_dataset("conll2004")

# Print some information about the dataset
print("Dataset structure:", dataset)
print("Sample data:", dataset['train'][0])