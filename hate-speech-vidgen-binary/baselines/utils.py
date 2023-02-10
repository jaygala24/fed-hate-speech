import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def iid_partition(data, clients):
    r""" Creates iid partitions of data over clients.
    """
    num_items_per_client = int(len(data) / clients)
    client_dict = {}
    data_idxs = list(range(len(data)))

    for i in range(clients):
        client_dict[i] = set(np.random.choice(data_idxs, num_items_per_client, replace=False))
        data_idxs = list(set(data_idxs) - client_dict[i])

    return client_dict


def pad_collate(batch):
    r""" Pads the sequence according to the max length in a mini batch.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_len = torch.tensor([len(x) for x in input_ids], dtype=torch.int64)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids_padded,
        "labels": torch.tensor(labels),
        "input_ids_len": input_ids_len,
    }
