import os
import warnings
import sklearn
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
#from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from loader.dataset import SentimentDataset
from transformers import *
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging
from model.model import SentimentClassifier
from loss.losses import FocalLoss
from sklearn.model_selection import StratifiedKFold
from transformers import get_constant_schedule
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def convert_lines(df, vocab, bpe, max_sequence_length):
    outputs = np.zeros((len(df), max_sequence_length))
    
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        subwords = bpe.encode('<s> '+row.content+' </s>')
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, default="./phobert/vocab.txt")
    parser.add_argument('--config_path', type=str, default="./phobert/config.json")
    parser.add_argument('--pretrained_path', type=str, required=True,default='./phobert/model.bin')
    parser.add_argument('--data_path', type=str, default='./data/data_segment.csv')
    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--bpe_codes', default="./phobert/bpe.codes",type=str, help='path to fastBPE BPE')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parser_opt()
    seed_everything(args.seed)
    bpe = fastBPE(args)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=True)
    df = pd.read_csv(args.data_path)
    df['content'] = df['content'].apply(str)
    vocab = Dictionary()
    vocab.add_from_file(args.dict_path)
    y = df.label.values
    X_train = convert_lines(df, vocab, bpe,args.max_sequence_length)
    
    
    config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_labels=1
)
    model = SentimentClassifier.from_pretrained(args.pretrained_path, config=config)
    model.cuda()

    criterion = FocalLoss(gamma=2, alpha=-0.25)
    # Recommendation by BERT: lr: 5e-5, 2e-5, 3e-5
    # Batchsize: 16, 32

    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs*len(X_train)/args.batch_size/args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
    scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler
    #optimizer = AdamW(model.parameters(), lr=args.lr)
    
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    best_acc = 0
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y))
    for fold, (train_idx, val_idx) in enumerate(splits):
        print("Training for fold {}".format(fold))
        best_score = 0
        if fold != args.fold:
            continue
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long), torch.tensor(y[train_idx],dtype=torch.long))
        valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(y[val_idx],dtype=torch.long))
        tq = tqdm(range(args.epochs + 1))
        for child in model.children():
            for param in child.parameters():
                if not param.requires_grad:
                    print("whoopsies")
                param.requires_grad = False
        frozen = True
        for epoch in tq:

            if epoch > 0 and frozen:
                for child in model.children():
                    for param in child.parameters():
                        param.requires_grad = True
                frozen = False
                del scheduler0
                torch.cuda.empty_cache()

            val_preds = None
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
            avg_loss = 0.
            avg_accuracy = 0.

            optimizer.zero_grad()

            pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
            for i,(x_batch, y_batch) in pbar:
                model.train()
                y_pred = model(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
                loss =  criterion(y_pred.view(-1).cuda(),y_batch.float().cuda())
                loss = loss.mean()
                loss.backward()
                if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    if not frozen:
                        scheduler.step()
                    else:
                        scheduler0.step()
                lossf = loss.item()
                pbar.set_postfix(loss = lossf)
                avg_loss += loss.item() / len(train_loader)
            model.eval()
            pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
            for i,(x_batch, y_batch) in pbar:
                y_pred = model(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
                y_pred = y_pred.squeeze().detach().cpu().numpy()
                val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
            val_preds = sigmoid(val_preds)

            best_th = 0
            score = f1_score(y[val_idx], val_preds > 0.5)
            print(f"\nAUC = {roc_auc_score(y[val_idx], val_preds):.4f}, F1 score @0.5 = {score:.4f}")
            if score >= best_score:
                torch.save(model.state_dict(),os.path.join(args.ckpt_path, f"model_{fold}.bin"))
                best_score = score