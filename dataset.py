import pickle
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torch import nn
from typing import List, Tuple


def load_file(file_path: str) -> Tuple[List[str], List[str]]:
    data = open(file_path, "r").readlines()
    for idx, val in enumerate(data):
        f, e = val.split("=")
        first_ = "<" + f + ">" + "="
        second_ = "<" + e + ">"
        data[idx] = first_ + second_
    return data


class Vocab:
    def __init__(self, data):
        unique_chars = set()
        for val in data:
            for char in val:
                unique_chars.add(char)
        self.vocab_mapping = {char: idx + 1 for idx, char in enumerate(unique_chars)}
        self.idx_pad_token = 0
        self.reverse_vocab_mapping = {v: k for k, v in self.vocab_mapping.items()}
        self.reverse_vocab_mapping[self.idx_pad_token] = ""
        self.idx_start_token = self.vocab_mapping["<"]
        self.idx_end_token = self.vocab_mapping[">"]

    def __len__(self):
        return len(self.vocab_mapping) + 1


class PreTrainedVocab:
    def __init__(self, vocab_mapping, reverse_vocab_mapping):
        with open('vocab_mapping.pkl', 'rb') as f:
            self.vocab_mapping = pickle.load(f)
        with open('reverse_vocab_mapping.pkl', 'rb') as f:
            self.reverse_vocab_mapping = pickle.load(f)
        self.idx_pad_token = 0
        self.idx_start_token = self.vocab_mapping["<"]
        self.idx_end_token = self.vocab_mapping[">"]

    def __len__(self):
        return len(self.vocab_mapping) + 1


def convert_tensor(vocab, sentence_input):
    x_idx = []
    x_idx = [vocab.vocab_mapping[x] for x in sentence_input]

    x_idx = torch.LongTensor(x_idx)
    return x_idx


def tensorify_data(vocab, input_data):
    tensor_input = []
    for input_data_sample in input_data:
        x_val = convert_tensor(vocab, input_data_sample)
        tensor_input.append(x_val)

    x_padded = nn.utils.rnn.pad_sequence(tensor_input, batch_first=True)

    return x_padded


class ExpressionDataset(Dataset):
    def __init__(self, path, split):
        self.input_data = load_file(path)
        if split == "train":
            self.input_data = self.input_data[: int(0.99 * len(self.input_data))]
        else:
            self.input_data = self.input_data[int(0.99 * len(self.input_data)) :]

        self.vocab = Vocab(self.input_data)
        factors, expansions = zip(
            *[line.strip().split("=") for line in self.input_data]
        )
        expansions = list(expansions)
        for idx, exp in enumerate(expansions):
            expansions[idx] = exp.replace("\n", "")
        self.x_padded = tensorify_data(self.vocab, factors)
        self.labels = tensorify_data(self.vocab, expansions)
        self.sep_pos = [sent.index("=") + 1 for sent in self.input_data]

    def __len__(self):
        return len(self.x_padded)

    def __getitem__(self, idx):
        return self.x_padded[idx], self.labels[idx]


ds = ExpressionDataset("train.txt", "validation")
