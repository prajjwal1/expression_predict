import torch
import math
from torch import nn
from torch.utils.data.dataloader import DataLoader
from dataset import ExpressionDataset
import pickle
from utils import train_one_epoch, validation_one_epoch
from model import Model

PATH = "train.txt"

DEVICE =  "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048
NUM_WORKERS = 4
LR = 2e-4

train_dataset = ExpressionDataset(PATH, split="train")
validation_dataset = ExpressionDataset(PATH, split="validation")
vocab = train_dataset.vocab

with open('vocab_mapping.pkl', 'wb') as f:
    pickle.dump(vocab.vocab_mapping, f)
with open('reverse_vocab_mapping.pkl', 'wb') as f:
    pickle.dump(vocab.reverse_vocab_mapping, f)


train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=1, num_workers=NUM_WORKERS
)

model = Model(train_dataset.vocab).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#  optimizer = Lamb(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(.9, .999))

criterion = nn.CrossEntropyLoss()

num_training_steps = len(train_dataset)//BATCH_SIZE
num_warmup_steps = int(num_training_steps*0.7)
num_cycles=0.5
num_epochs = 452

def lr_lambda(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

for epoch in range(num_epochs):
    model, optimizer, lr_scheduler = train_one_epoch(
        train_dataloader, model, optimizer, lr_scheduler, criterion, device=DEVICE
    )
    if (epoch+1) % 20  == 0:
        accuracy = validation_one_epoch(validation_dataloader, model, vocab, device=DEVICE)
        print("Accuracy on Epoch ", epoch, ":", accuracy)
        torch.save(model.state_dict(), "model.pth")
        torch.save(optimizer.state_dict(), "optimizer.pth")
        torch.save(lr_scheduler.state_dict(), "scheduler.pth")

