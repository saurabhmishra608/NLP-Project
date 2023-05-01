import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import BertModel, BertTokenizer,BertForSequenceClassification,AutoModel,BertTokenizerFast,AutoTokenizer



# # specify GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # import BERT-base pretrained model
# bert = AutoModel.from_pretrained('roberta-base')

# # Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

max_seq_len = 50

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      self.lstm = nn.LSTM(768, 512)
      self.linear = nn.Linear(512*2, 256)
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 
      self.fc2 = nn.Linear(512,256)

      # dense layer 3 (Output Layer)
      self.fc3 = nn.Linear(256,7)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask,return_dict=False)
      
      #lstm_output, (h,c) = self.lstm(a) ## extract the 1st token's embeddings
      #hidden = torch.cat((lstm_output[:,-1, :512],lstm_output[:,0, 512:]),dim=-1)
      #linear_output = self.linear(lstm_output[:,-1].view(-1,512*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification
      
      #x=linear_output

      x, _ = self.lstm(cls_hs)
      #x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # 2nd dense layer
      x = self.fc2(x)

      x = self.relu(x)

      x = self.dropout(x)
      
      # output layer
      self.fc3(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x

# pass the pre-trained BERT to our define architecture
# model = BERT_Arch(bert)

# #load weights of best model
# path = '/home/saurabh/dig_path/RISE_CAM16/yolo-hand-detection-master/models/BERT.pt'
# model.load_state_dict(torch.load(path))

def classify_intent(input_sentence, model, max_seq_len=50):
  tokens = tokenizer.batch_encode_plus(
    input_sentence,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
  )
  labelMap = {0: "AddToPlaylist",1: "BookRestaurant",2: "GetWeather",3: "PlayMusic",4: "RateBook",5: "SearchCreativeWork",6: "SearchScreeningEvent"}
  seq = torch.tensor(tokens['input_ids'])
  mask = torch.tensor(tokens['attention_mask'])

  model.to(device)
  model.eval()

  with torch.no_grad():
    preds = model(seq.to(device), mask.to(device))
    preds = preds.detach().cpu().numpy()

  preds = np.argmax(preds, axis = 1)

  return labelMap[preds[0]]

# input_sentence = ["PLAY U"]
# print("The classified intent is: ",classify_intent(input_sentence, model, max_seq_len=50))
