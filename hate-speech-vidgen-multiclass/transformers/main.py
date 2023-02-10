import os
import sys
import time
import warnings
import logging
import argparse
import wandb
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)
from data import *
from trainer import *
from utils import *

# command-line arguments
parser = argparse.ArgumentParser("hate speech classification using federated learning")
parser.add_argument("--data", type=str, default="data", help="location of the data corpus")
parser.add_argument(
    "--batch_size", type=int, default=64, help="batch size for fine-tuning the model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for deterministic behaviour and reproducibility",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="distilbert",
    help="specify which model to use (fnet, distilbert, bert, distilroberta, roberta)",
)
parser.add_argument("--rounds", type=int, default=1000, help="number of training rounds")
parser.add_argument("--C", type=float, default=0.1, help="client fraction")
parser.add_argument("--K", type=int, default=100, help="number of clients for iid partition")
parser.add_argument(
    "--E",
    "--epochs",
    type=int,
    default=1,
    help="number of training epochs on local dataset for each round",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="fedprox",
    help="specify which algorithm to use during local updates aggregation (fedopt, fedprox, fedavg)",
)
parser.add_argument("--mu", type=float, default=0.01, help="proximal term constant")
parser.add_argument("--client_lr", type=float, default=2e-5, help="learning rate for client")
parser.add_argument("--server_lr", type=float, default=0.0, help="learning rate for server")
parser.add_argument(
    "--class_weights",
    action="store_true",
    default=False,
    help="determine if experiments should use class weights in loss function or not",
)
parser.add_argument("--es_patience", type=int, default=10, help="early stopping patience level")
parser.add_argument("--save", type=str, default="exp", help="experiment name")
parser.add_argument(
    "--wandb", action="store_true", default=False, help="use wandb for tracking results",
)
parser.add_argument(
    "--wandb_proj_name",
    type=str,
    default="hate-speech-vidgen-multiclass",
    help="provide a project name",
)
parser.add_argument("--wandb_run_name", type=str, default="exp", help="provide a run name")
parser.add_argument(
    "--wandb_run_notes", type=str, default="", help="provide notes for a run if any",
)
args = parser.parse_args()

# proximal term is 0.0 in case of fedavg
if args.algorithm == "fedavg":
    args.mu = 0.0

# we don't need server optimizer in case of fedprox and fedavg + centralized training
if args.algorithm != "fedopt" or args.K == 1:
    args.server_lr = None
args.algorithm = None if args.K == 1 else args.algorithm

args.save = "{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))

if not os.path.exists(args.save):
    os.mkdir(args.save)
print("Experiment dir: {}".format(args.save))


log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

warnings.filterwarnings("ignore")


def main():
    if not torch.cuda.is_available():
        logging.info("No GPU device available")
        sys.exit(1)

    # log all the arguments
    logging.info(args)

    # reproducibility set seed
    set_seed(args.seed)

    raw_data = {
        "train": pd.read_csv(os.path.join(args.data, "train.csv")),
        "valid": pd.read_csv(os.path.join(args.data, "valid.csv")),
        "test": pd.read_csv(os.path.join(args.data, "test.csv")),
    }

    # ignore abusive category due to insufficient examples
    for split in raw_data:
        raw_data[split] = raw_data[split][raw_data[split]["category"] != "abusive"]
        raw_data[split].reset_index(inplace=True, drop=True)

    # load the tokenizer from huggingface hub
    model_ckpt = get_model_ckpt(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    new_categories = [cat for cat in categories if cat != "abusive"]
    category2id = {cat: idx for idx, cat in enumerate(new_categories)}
    id2category = {idx: cat for idx, cat in enumerate(new_categories)}
    args.class_names = new_categories

    dataset = {
        "train": create_data_iter(raw_data["train"], category2id, tokenizer),
        "valid": create_data_iter(raw_data["valid"], category2id, tokenizer),
        "test": create_data_iter(raw_data["test"], category2id, tokenizer),
    }

    # categorical classes for hate speech data
    classes = list(np.unique([elem["labels"].item() for elem in dataset["train"]]))
    class_array = np.array([elem["labels"].item() for elem in dataset["train"]])
    num_classes = len(classes)
    args.classes = classes
    args.num_classes = num_classes

    if args.class_weights:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=classes, y=class_array
        )
    else:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=classes, y=classes
        )
    args.class_weights_array = class_weights
    logging.info(f"# of classes: {num_classes}")
    logging.info(f"class weights: {class_weights}")

    # load the model from huggingface hub
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=num_classes, label2id=category2id, id2label=id2category,
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model.cuda()

    # dict mapping clients to the data samples in iid fashion
    iid_data_dict = iid_partition(dataset["train"], args.K)

    # log the config to the wandb
    config_dict = dict(
        rounds=args.rounds,
        C=args.C,
        K=args.K,
        E=args.E,
        model=args.model_type,
        algorithm=args.algorithm,
        mu=args.mu,
        client_lr=args.client_lr,
        server_lr=args.server_lr,
        batch_size=args.batch_size,
        class_weights=args.class_weights,
    )

    if args.wandb:
        run = wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_proj_name,
            notes=args.wandb_run_notes,
            config=config_dict,
        )

    fl_trainer = FedTrainer(
        args,
        tokenizer,
        model,
        local_data_idxs=iid_data_dict,
        train_data=dataset["train"],
        val_data=dataset["valid"],
        test_data=dataset["test"],
    )
    model = fl_trainer.train()

    if args.wandb:
        artifact = wandb.Artifact(args.wandb_run_name, type="model")
        artifact.add_dir(args.save)
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
