# Fed Hate Speech

The official code repository associated with the paper titled "A Federated Approach for Hate Speech Detection" (EACL 2023).

## Overview

> Hate speech detection has been the subject of high research attention, due to the scale of content created on social media. In spite of the attention and the sensitive nature of the task, privacy preservation in hate speech detection has remained under-studied. The majority of research has focused on centralised machine learning infrastructures which risk leaking data. In this paper, we show that using federated machine learning can help address privacy the concerns that are inehrent to hate speech detection while obtaining up to 6.81% improvement in terms of F1-score.

## Installation

Instructions to setup and install everything before running the code.

```
<!-- Clone the github repository and navigate to the project directory. -->
git clone https://github.com/jaygala24/fed-hate-speech.git
cd fed-hate-speech

<!-- Install all the dependencies and requirements associated with the project. -->
pip3 install -r requirements.txt
```

Note: We recommend creating a virtual environment to run the experiments.


## Data Preparation

Please refer to the paper for the complete details on the schema normalization and data pre-processing steps for different datasets.


## Running Experiments

We experiment with two federated learning algorithm (`FedProx` and `FedOPT`) and different model (`Logistic Regression`, `LSTM`, `DistilBERT`, `FNet` and `RoBERTa`) variants. Please refer to the instructions below to run different experiments.

### FedProx Variant

In order to train the federated transformer-based `DistilBERT` model using `FedProx` algorithm for a client fraction of 10% and 5 local epochs on the combined dataset, run the following command in the `transformers` directory:

```
python3 main.py --data data --dataset_type comb --rounds 50 --C 0.1 --E 5 --K 100 \
                --algorithm fedprox --mu 0.01 --client_lr 4e-5 --server_lr 0.0 \
                --model distilbert --batch_size 32 --seed 42 --class_weights \
                --save distilbert_fedprox_c0.1_e05_k100_r50_class_weighted
```

In order to train the federated baseline `LSTM` model using `FedProx` algorithm for a client fraction of 10% and 5 local epochs on the combined dataset, run the following command in the `baselines` directory:

```
python3 main.py --data data --dataset_type comb --rounds 50 --C 0.1 --E 5 --K 100 \
                --algorithm fedprox --mu 0.01 --client_lr 1e-3 --server_lr 0.0 \
                --model distilbert --batch_size 128 --seed 42 --class_weights \
                --save lstm_fedprox_c0.1_e05_k100_r50_class_weighted
```

### FedOPT Variant 

In order to train the federated transformer-based `DistilBERT` model using `FedOPT` algorithm for a client fraction of 10% and 5 local epochs on the combined dataset, run the following command in the `transformers` directory:

```
python3 main.py --data data --dataset_type comb --rounds 50 --C 0.1 --E 5 --K 100 \
                --algorithm fedopt --mu 0.0 --client_lr 4e-5 --server_lr 1e-3 \
                --model distilbert --batch_size 32 --seed 42 --class_weights \
                --save distilbert_fedopt_c0.1_e05_k100_r50_class_weighted
```

In order to train the federated baseline `LSTM` model using `FedProx` algorithm for a client fraction of 10% and 5 local epochs on the combined dataset, run the following command in the `baselines` directory:

```
python3 main.py --data data --dataset_type comb --rounds 50 --C 0.1 --E 5 --K 100 \
                --algorithm fedopt --mu 0.0 --client_lr 1e-3 --server_lr 1e-2 \
                --model distilbert --batch_size 128 --seed 42 --class_weights \
                --save lstm_fedopt_c0.1_e05_k100_r50_class_weighted
```

<br>

In order to run the above algorithm variants for different client fractions (`C`) and local epochs (`E`), please change the arguments `--C` to 0.1, 0.3 and 0.5 and `--E` to 1, 5 and 20 respectively. Please refer to the paper appendix for other hyperparameters such as batch size (`bs`), client learning (`client_lr`), server learning rate (`server_lr`) and proximal term (`mu`) for different model and algorithm variants.

Similarly, you can run experiments on "Learning from the Worst" dataset ([Vidgen et al., 2021](https://aclanthology.org/2021.acl-long.132)) by changing the `--dataset_type` to either `vidgen_binary` (for binary classification) or `vidgen_multiclass` (for multiclass classification).

Important Arguments:
- `dataset_type`: which dataset to use for experiments
- `model_type`: different model variants to use for experiments
- `rounds`: number of federated training rounds
- `C`: client fraction for each federated training round
- `K`: number of clients for iid partition
- `E`: number of local training epochs on local dataset for each round
- `algorithm`: local client updates aggregation strategy to follow on server
- `mu`: proximal term constant acting as regularizer to ensure the local client updates to be similar to global model
- `client_lr`: learning rate on client devices
- `server_lr`: learning rate on server

Note: The models can be trained centrally by setting the `--C` parameter to 1.0 and `--K` to 1.


## Todos

- [x] Add data preparation instruction in the README
- [ ] Add LICENSE in the README
- [ ] Add citation bibtex in the README
