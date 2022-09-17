import os
import sys
import time
import copy
import warnings
import logging
import argparse
import wandb
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AdamW,
    set_seed,
)
from data import *
from utils import *

# command-line arguments
parser = argparse.ArgumentParser("hate speech classification using federated learning")
parser.add_argument("--data", type=str, default="data", help="location of the data corpus")
parser.add_argument(
    "--batch_size", type=int, default=64, help="batch size for fine-tuning the model"
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
    "--percentage", type=float, default=0, help="percentage of clients to have fewer than E epochs"
)
parser.add_argument(
    "--class_weights",
    action="store_true",
    default=False,
    help="determine if experiments should use class weights in loss function or not",
)
parser.add_argument("--es_patience", type=int, default=10, help="early stopping patience level")
parser.add_argument("--save", type=str, default="exp", help="experiment name")
parser.add_argument(
    "--wandb", action="store_true", default=False, help="use wandb for tracking results"
)
parser.add_argument(
    "--wandb_proj_name", type=str, default="hate-speech-vidgen-multiclass", help="provide a project name"
)
parser.add_argument("--wandb_run_name", type=str, default="exp", help="provide a run name")
parser.add_argument(
    "--wandb_run_notes", type=str, default="", help="provide notes for a run if any"
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
        model_ckpt, num_labels=num_classes, label2id=category2id, id2label=id2category
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
        percentage=args.percentage,
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


class FedTrainer:
    r""" Implements fine-tuning of transformer models in federated learning setup.
         Currently supports the following aggregation strategy for local model updates.
         - FedAdam (FedOpt)
         - FedProx
         - FedAvg
    """

    def __init__(self, args, tokenizer, model, local_data_idxs, train_data, val_data, test_data):
        self.args = args
        self.num_clients = max(int(args.C * args.K), 1)
        self.tokenizer = tokenizer
        self.model = model
        self.local_data_idxs = local_data_idxs
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # initialize the server optimizer according to the aggregation algorithm
        self._init_opt()

    def _init_opt(self):
        # we don't need server optimizer in case of fedprox and fedavg + centralized training
        if self.args.K == 1 or self.args.algorithm != "fedopt":
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.server_lr)

    def _generate_local_epochs(self):
        r""" Generates a list of epochs for selected clients to replicate system heterogeneity
        """
        if self.args.percentage == 0.0:
            # each client runs for the same E epochs
            return np.array([self.args.E] * self.num_clients)
        else:
            # get the number of clients to have fewer than E epochs
            heterogenous_size = int((self.args.percentage / 100) * self.num_clients)
            # generate random uniform epochs of heterogenous size between 1 and E
            client_epoch_list = np.random.randint(1, self.args.E, heterogenous_size)

            # the rest of the clients will have E epochs
            rem_clients = self.num_clients - heterogenous_size
            rem_client_list = [self.args.E] * rem_clients

            # aggregate the heterogenous and non-heterogenous list of epochs
            client_epoch_list = np.append(client_epoch_list, rem_client_list, axis=0)
            # shuffle the list and return
            np.random.shuffle(client_epoch_list)
            logging.info("local epochs generation done")
            return client_epoch_list

    def train(self):
        best_f1_loss = float("-inf")
        self.args.es_patience
        patience = 0

        logging.info(f"System heterogeneity set to {self.args.percentage}% stragglers.")
        logging.info(f"Picking {self.num_clients} random clients per round.")

        for round_idx in range(1, self.args.rounds + 1):
            w_locals, local_loss = [], []
            client_epoch_list = self._generate_local_epochs()
            client_idxs = np.random.choice(range(self.args.K), self.num_clients, replace=False)

            if self.args.algorithm == "fedavg":
                stragglers_idxs = np.argwhere(client_epoch_list < self.args.E)
                client_epoch_list = np.delete(client_epoch_list, stragglers_idxs)
                client_idxs = np.delete(client_idxs, stragglers_idxs)

            for client_idx, epoch in zip(client_idxs, client_epoch_list):
                client = Client(
                    self.args,
                    epoch,
                    self.train_data,
                    self.local_data_idxs[client_idx],
                    self.tokenizer,
                )
                w, loss = client.train(model=copy.deepcopy(self.model))
                w_locals.append(copy.deepcopy(w))
                local_loss.append(loss)

            # updating the global weights
            w_avg = copy.deepcopy(w_locals[0])
            for k in w_avg.keys():
                for i in range(1, len(w_locals)):
                    w_avg[k] += w_locals[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w_locals))

            if self.optimizer == None:
                # no optimization in case of fedprox and fedavg + centralized training
                self.model.load_state_dict(w_avg)
            else:
                # aggregate the local updates using server optimizer in case of fedopt
                self.optimizer.zero_grad()
                optimizer_state = self.optimizer.state_dict()
                # set model global grads
                new_model = copy.deepcopy(self.model)
                new_model.load_state_dict(w_avg)
                with torch.no_grad():
                    for parameter, new_parameter in zip(
                        self.model.parameters(), new_model.parameters()
                    ):
                        parameter.grad = parameter.data - new_parameter.data
                        # because we go to the opposite direction of the gradient
                model_state_dict = self.model.state_dict()
                new_model_state_dict = new_model.state_dict()
                for k in dict(self.model.named_parameters()).keys():
                    new_model_state_dict[k] = model_state_dict[k]
                self.model.load_state_dict(new_model_state_dict)
                # instantiate the optimizer and copy the optimizer state tracked above
                self._init_opt()
                self.optimizer.load_state_dict(optimizer_state)
                self.optimizer.step()
                del new_model   # to avoid GPU OOM issue

            avg_train_loss = sum(local_loss) / len(local_loss)

            # evaluate the model on validation set
            val_metrics, val_loss = self.eval(stage="valid")
            logging.info(
                f"Round: {round_idx}... \tAverage Train Loss: {round(avg_train_loss, 3)}... \tDev Loss: {round(val_loss, 3)}... "
                f"\tDev Accuracy: {val_metrics['valid/accuracy']}... \tAUC Score: {val_metrics['valid/auc']}... \tPrecision: {val_metrics['valid/precision']}... "
                f"\tRecall: {val_metrics['valid/recall']}... \tF1: {val_metrics['valid/f1_score']}... \tMCC: {val_metrics['valid/mcc']}"
            )

            test_metrics = {
                "test/accuracy": 0.0,
                "test/auc": 0.0,
                "test/precision": 0.0,
                "test/recall": 0.0,
                "test/f1_score": 0.0,
                "test/mcc": 0.0,
            }
            test_loss = 0.0

            # early stop if we don't improve till patience level
            if val_metrics["valid/f1_score"] > best_f1_loss:
                logging.info(
                    f"Dev f1 score improved ({best_f1_loss:.4f} -> {val_metrics['valid/f1_score']:.4f}). Saving model!"
                )
                best_f1_loss = val_metrics["valid/f1_score"]
                patience = 0
                # save the model and tokenizer
                if torch.cuda.device_count() > 1:
                    self.model.module.save_pretrained(self.args.save)
                else:
                    self.model.save_pretrained(self.args.save)
                self.tokenizer.save_pretrained(self.args.save)
            else:
                patience += 1
                logging.info(f"Early stopping counter {patience} out of {self.args.es_patience}")
                if patience == self.args.es_patience:
                    # load the model and tokenizer with best performance
                    if torch.cuda.device_count() > 1:
                        self.model.module.from_pretrained(self.args.save)
                    else:
                        self.model.from_pretrained(self.args.save)
                    self.tokenizer.from_pretrained(self.args.save)
                    # evaluate the model on test set
                    test_metrics, test_loss = self.eval(stage="test")
                    logging.info(
                        f"FINAL TESTING\n... \tTest Loss: {round(test_loss, 3)}... Test Accuracy: {test_metrics['test/accuracy']}...  "
                        f"\tAUC Score: {test_metrics['test/auc']}... \tPrecision: {test_metrics['test/precision']}... \tRecall: {test_metrics['test/recall']}... "
                        f"\tF1: {test_metrics['test/f1_score']}... \tMCC: {test_metrics['test/mcc']}"
                    )
                    if self.args.wandb:
                        wb_metrics = {
                            "train/loss": avg_train_loss,
                            "valid/loss": val_loss,
                            **val_metrics,
                            "test/loss": test_loss,
                            **test_metrics,
                        }
                        wandb.log(wb_metrics, step=round_idx)
                        wandb.run.summary["valid/f1_score"] = val_metrics["valid/f1_score"]
                    break

            # finally evaluate the model on the test set
            if round_idx == self.args.rounds:
                # load the model and tokenizer with best performance
                if torch.cuda.device_count() > 1:
                    self.model.module.from_pretrained(self.args.save)
                else:
                    self.model.from_pretrained(self.args.save)
                self.tokenizer.from_pretrained(self.args.save)
                # evaluate the model on test set
                test_metrics, test_loss = self.eval(stage="test")
                logging.info(
                    f"FINAL TESTING\n... \tTest Loss: {round(test_loss, 3)}... Test Accuracy: {test_metrics['test/accuracy']}...  "
                    f"\tAUC Score: {test_metrics['test/auc']}... \tPrecision: {test_metrics['test/precision']}... \tRecall: {test_metrics['test/recall']}... "
                    f"\tF1: {test_metrics['test/f1_score']}... \tMCC: {test_metrics['test/mcc']}"
                )
            if self.args.wandb:
                wb_metrics = {
                    "train/loss": avg_train_loss,
                    "valid/loss": val_loss,
                    **val_metrics,
                    "test/loss": test_loss,
                    **test_metrics,
                }
                wandb.log(wb_metrics, step=round_idx)
                wandb.run.summary["valid/f1_score"] = val_metrics["valid/f1_score"]

        return self.model

    def eval(self, stage="valid"):
        assert stage in ("valid", "test"), f"stage: {stage} not supported"
        data = self.val_data if stage == "valid" else self.test_data
        dataloader = DataLoader(
            data,
            batch_size=self.args.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.args.class_weights_array).cuda()
        )

        self.model.eval()
        pred_probs, labels = [], []

        batch_loss = []
        for batch in dataloader:
            # transfer the data (tensors) to GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch)
                logits = output.logits
                loss = criterion(logits, batch["labels"])
                batch_loss.append(loss.item())

            # gather the predictions and labels which will be used later for calculating metrics
            probs = torch.softmax(logits, dim=-1)
            pred_probs.append(probs)
            labels.append(batch["labels"])

        epoch_loss = sum(batch_loss) / len(batch_loss)

        pred_probs = torch.cat(pred_probs)
        preds = torch.argmax(pred_probs, dim=-1)
        labels = torch.cat(labels)

        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        if self.args.num_classes < 3:
            auc = roc_auc_score(
                labels.cpu().numpy(),
                pred_probs.cpu().numpy()[:, 1],
                labels=self.args.classes,
            )
        else:
            auc = roc_auc_score(
                labels.cpu().numpy(),
                pred_probs.cpu().numpy(),
                multi_class="ovo",
                labels=self.args.classes,
            )
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            labels.cpu().numpy(), 
            preds.cpu().numpy(), 
            average="weighted" if self.args.num_classes > 2 else "macro", 
            labels=self.args.classes
        )
        mcc = matthews_corrcoef(labels.cpu().numpy(), preds.cpu().numpy())

        metrics = {
            f"{stage}/accuracy": 100.0 * accuracy,
            f"{stage}/auc": auc,
            f"{stage}/precision": precision,
            f"{stage}/recall": recall,
            f"{stage}/f1_score": f1_score,
            f"{stage}/mcc": mcc,
        }

        return metrics, epoch_loss


class Client:
    def __init__(self, args, epochs, data, idxs, tokenizer):
        self.args = args
        self.epochs = epochs
        self.dataloader = DataLoader(
            CustomDataset(data, idxs),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
        )

    def train(self, model):
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.args.class_weights_array).cuda()
        )
        optimizer = AdamW(
            model.parameters(),
            lr=self.args.client_lr,
            weight_decay=0.01 if self.args.model_type == "fnet" else 0,
        )

        # use the weights of global model for proximal term calculation
        global_model = copy.deepcopy(model)

        model.train()

        epoch_loss = []
        for epoch in range(1, self.epochs + 1):
            batch_loss = []
            for batch in self.dataloader:
                # transfer the data (tensors) to GPU
                batch = {k: v.cuda() for k, v in batch.items()}

                # flush the gradients and perform a forward pass
                optimizer.zero_grad()
                output = model(**batch)
                logits = output.logits

                # calculate the loss
                # proximal term in fedprox acts as a kind of L2 regularization
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                # don't include the proximal term for central training
                if int(self.args.K) == 1:
                    loss = criterion(logits, batch["labels"])
                else:
                    loss = criterion(logits, batch["labels"]) + (self.args.mu / 2) * proximal_term

                # perform the backward pass and update the local model parameters
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        total_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), total_loss


if __name__ == "__main__":
    main()
