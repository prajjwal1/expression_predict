import sys
import numpy as np
from os.path import exists
from typing import Tuple

from model import Model
from dataset import PreTrainedVocab, tensorify_data
from utils import standardize_input, put_on_device
import torch
from torch import nn

TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str, model: nn.Module, vocab: PreTrainedVocab):
    factors = "<" + factors + ">" + "="

    x_padded = tensorify_data(vocab, factors)
    x_padded = put_on_device(x_padded, device=DEVICE)[0]
    x_padded = x_padded.permute(1, 0)
    output = model.predict(x_padded)
    output = output.detach().squeeze(1).tolist()  # [seq_len, bs]

    decoded_pred = "".join(
            [vocab.reverse_vocab_mapping[token_id] for token_id in output]
        )

    return standardize_input(decoded_pred)


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    if  "test.txt" not in filepath:
        print("PLEASE ENSURE THAT 'test.txt' is present in this directory")
        print("If 'test.txt' is present, please pass '-t' as arg (python3 main.py -t)")
        print("INFERENCE WILL NOW RUN ON PROVIDED TRAINING SET\n")

    factors, expansions = load_file(filepath)

    vocab_mapping_path = "vocab_mapping.pkl"
    reverse_vocab_mapping_path = "reverse_vocab_mapping.pkl"
    model_weights_path = "model.pth"

    vocab = PreTrainedVocab(vocab_mapping_path, reverse_vocab_mapping_path)
    model = Model(vocab).to(device=DEVICE)
    model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
    model.eval()

    pred = [predict(f, model, vocab) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")
