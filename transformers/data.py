import torch

# categories for combined dataset (multiclass setup)
comb_data_categories = [
    "none",
    "toxicity",
    "offensive",
    "hate_speech",
    "misogyny_sexism",
    "aggressive_hate_speech",
    "insult",
    "severe_toxic",
    "threat",
    "aggression",
    "covert_aggression",
    "overt_aggression",
    "racism",
    "abusive",
]

# categories for vidgen dataset (binary setup)
vidgen_binary_data_categories = ["hate", "nothate"]

# categories for vidgen dataset (multiclass setup)
vidgen_multiclass_data_categories = [
    "none",
    "notgiven",
    "derogation",
    "animosity",
    "dehumanization",
    "threatening",
    "support",
]


def get_categories(dataset_type):
    if dataset_type == "comb":
        return comb_data_categories
    elif dataset_type == "vidgen_binary":
        return vidgen_binary_data_categories
    elif dataset_type == "vidgen_multiclass":
        return vidgen_multiclass_data_categories
    else:
        raise NotImplementedError


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
