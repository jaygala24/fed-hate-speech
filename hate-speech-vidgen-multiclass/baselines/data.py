import torch
from torchtext.vocab import build_vocab_from_iterator

categories = [
    'none', 
    'notgiven', 
    'derogation', 
    'animosity', 
    'dehumanization', 
    'threatening', 
    'support'
]


def create_data_iter(df, category_dict, input_col="text", target_col="category"):
    r""" Creates data iterator as list of tuple consisting of `text` and `category`.
    """
    # maps category to the corresponding integer
    df[target_col] = df[target_col].apply(lambda x: category_dict[x])
    # iterate over the data and tokenize the samples
    iterator = []
    for i in range(len(df)):
        iterator.append({"text": df[input_col].iloc[i].lower(), "label": df[target_col].iloc[i]})
    return iterator


def build_vocab(iterator, tokenizer):
    r""" Builds a vocabulary with tokens from the data iterator.
    """

    def yield_tokens(iterator, tokenizer):
        for item in iterator:
            yield tokenizer(item["text"])

    vocab = build_vocab_from_iterator(yield_tokens(iterator, tokenizer), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def create_final_dataset(iterator, vocab, tokenizer):
    r""" Pipeline for processing the raw text into a tokenized text.
    """
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    dataset = []
    for item in iterator:
        dataset.append(
            {
                "input_ids": torch.tensor(text_pipeline(item["text"]), dtype=torch.int64),
                "labels": torch.tensor(label_pipeline(item["label"])),
            }
        )
    return dataset


class CustomDataset:
    r""" Custom dataset wrapper for hate speech data.
    """

    def __init__(self, data, idxs):
        self.data = data
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.data[self.idxs[item]]
