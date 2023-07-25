import torch
import torch.nn as nn
from transformers import AutoModel


class SentimentClassifier(nn.Module):
  def __init__(self, n_classes=2):
    super(SentimentClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained('vinai/phobert-base')
    self.drop = nn.Dropout(p=0.5)
    #self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.qa_outputs = nn.Linear(4*self.bert.config.hidden_size, n_classes)
    nn.init.normal_(self.qa_outputs.weight, std=0.02)
    nn.init.normal_(self.qa_outputs.bias,0)
  
  def forward(self, input_ids, attention_mask):
    last_hidden_state, output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask,
        return_dict=False
    )
    cls_output = torch.cat((output[2][-1][:,0, ...],output[2][-2][:,0, ...],output[2][-3][:,0, ...], output[2][-4][:,0, ...]),-1)
    logits = self.qa_outputs(cls_output)

    return logits