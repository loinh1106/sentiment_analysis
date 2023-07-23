import warnings
import sklearn
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from loader.dataset import SentimentDataset
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging
from model.model import SentimentClassifier
from loss.losses import FocalLoss

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def prepare_loaders(df_train,df_val, tokenizer):
    
    train_dataset = SentimentDataset(df_train, tokenizer, max_len=120)
    valid_dataset = SentimentDataset(df_val, tokenizer, max_len=120)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    return train_loader, valid_loader

def train(model, criterion, optimizer, train_loader,lr_scheduler):
    model.train()
    losses = []
    correct = 0

    for data in train_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, targets)
        _, pred = torch.max(outputs, dim=1)

        correct += torch.sum(pred == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

    print(f'Train Accuracy: {correct.double()/len(train_loader.dataset)} Loss: {np.mean(losses)}')

def eval(model, test_loader,criterion,test_data = False):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        data_loader = test_loader if test_data else valid_loader
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)
            correct += torch.sum(pred == targets)
            losses.append(loss.item())
    
    if test_data:
        print(f'Test Accuracy: {correct.double()/len(test_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(test_loader.dataset)
    else:
        print(f'Valid Accuracy: {correct.double()/len(valid_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(valid_loader.dataset)


if __name__ == '__main__':
    seed_everything(86)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2', use_fast=True)
    df = pd.read_csv('./data/data_segment.csv')
    X_train,X_test, y_train, y_test = train_test_split(df, df['label'],test_size = 0.1, random_state= 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.1, random_state= 42)
    train_loader, valid_loader = prepare_loaders(X_train, X_val, tokenizer=tokenizer)
    testset = SentimentDataset(X_test, tokenizer=tokenizer)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False)
    model = SentimentClassifier(n_classes=2).to(device)
    criterion = FocalLoss(gamma=2, alpha=-0.25)
    # Recommendation by BERT: lr: 5e-5, 2e-5, 3e-5
    # Batchsize: 16, 32
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=len(train_loader)*1
            )
    best_acc = 0
    for epoch in range(2):
        print(f'Epoch {epoch+1}/{1}')
        print('-'*30)

        train(model, criterion, optimizer, train_loader, lr_scheduler=lr_scheduler)
        val_acc = eval(model=model,criterion=criterion, test_loader=test_loader)

        if val_acc > best_acc:
            torch.save(model.state_dict(), f'./ckpt/phobert_at_{epoch}_epoch.pth')
            best_acc = val_acc