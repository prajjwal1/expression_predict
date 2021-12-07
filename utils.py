from tqdm import tqdm
import torch
from typing import List

def put_on_device(*args, device):
    all_args = []
    for arg in args:
        all_args.append(arg.to(device=device))
    return all_args


def compute_accuracy(true_expansion: List[str], pred_expansion: List[str]) -> float:
    """the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    correct = 0
    for gt, pred in zip(true_expansion, pred_expansion):
        if gt == pred:
            correct += 1
    return correct / len(true_expansion)


def standardize_input(sentence: str) -> str:
    replace_ops = [(" ", ""), ("<", ""), (">", "")]
    for k, v in replace_ops:
        sentence = sentence.replace(k, v)
    sentence = sentence.strip()
    return sentence


def train_one_epoch(
    train_dataloader, model, optimizer, lr_scheduler, criterion, device
):
    train_pbar = tqdm(enumerate(train_dataloader))

    for num_step, data in train_pbar:
        model.train()
        inputs, labels = data

        inputs, labels = put_on_device(inputs, labels, device=device)

        teach_forcing_tokens = labels.clone()
        teach_forcing_tokens = put_on_device(teach_forcing_tokens, device=device)[0]

        optimizer.zero_grad()
        outputs = model(inputs, teach_forcing_tokens)

        loss = criterion(outputs[:, :-1, :].permute(0, 2, 1), labels[:, 1:])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_pbar.set_postfix({"loss": loss})
    return model, optimizer, lr_scheduler


def validation_one_epoch(validation_dataloader, model, vocab, device):
    validation_pbar = tqdm(enumerate(validation_dataloader))
    model.eval()
    print("\nRunning Evaluation")
    gt, preds = [], []
    for valid_step, valid_data in validation_pbar:
        with torch.no_grad():
            valid_input, valid_label = valid_data

            valid_input = put_on_device(valid_input, device=device)[0]
            pred = model.predict(valid_input)
            pred = pred.cpu().detach().squeeze(1).tolist()  # [seq_len, bs]

            decoded_pred = "".join(
                [vocab.reverse_vocab_mapping[token_id] for token_id in pred]
            )
            preds.append(standardize_input(decoded_pred))

            valid_label = valid_label.cpu().squeeze(0).tolist()  # [bs, seq_len]
            decoded_label = "".join(
                [vocab.reverse_vocab_mapping[token_id] for token_id in valid_label]
            )
            gt.append(standardize_input(decoded_label))
    return compute_accuracy(gt, preds)
