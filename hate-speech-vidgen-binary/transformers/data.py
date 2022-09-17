import torch

categories = [
    "hate", 
    "nothate"
]

def create_data_iter(df, category_dict, tokenizer, input_col="text", target_col="category"):
    r""" Creates data iterator as list of tuple consisting of `text` and `category`.
    """
    # maps category to the corresponding integer
    df[target_col] = df[target_col].apply(lambda x: category_dict[x])
    # iterate over the data and tokenize the samples
    iterator = []
    for i in range(len(df)):
        enc_inputs = tokenizer(df[input_col].iloc[i].lower(), truncation=True, padding=False)
        enc_inputs = {k: torch.tensor(v) for k, v in enc_inputs.items()}
        iterator.append(
            {**enc_inputs, "labels": torch.tensor(df[target_col].iloc[i]),}
        )
    return iterator


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
