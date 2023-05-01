from Sentence import getSentence
from videoProcess import videoProcess
from intentClassify import classify_intent
from intentClassify import BERT_Arch
import transformers
from transformers import BertModel, BertTokenizer,BertForSequenceClassification,AutoModel,BertTokenizerFast,AutoTokenizer
import torch
import warnings
import logging
import argparse
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Process video data using a CNN.')
parser.add_argument('videoInput', type=str, help='Path to the input video file')
args = parser.parse_args()
videoInput = args.videoInput

device = 'cuda' if torch.cuda.is_available() else 'cpu'

outputPath = 'output_video.avi'
#videoInput = "input.mp4"
cnnPath = 'models/hand_rec2.pt'


charSeq = videoProcess(videoPath = videoInput, outputPath = outputPath, cnnPath = cnnPath,network = "prn",device = 0,size = 416,confidence = 0.2,hands = 1)
print("Extracted sequence = ",charSeq)
#charSeq = "playu"
sentence = getSentence(charSeq)
print("Extracted sentence = ",sentence)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
max_seq_len = 50
bert = AutoModel.from_pretrained('roberta-base')
model = BERT_Arch(bert)

#load weights of best model
path = 'models/BERT.pt'
model.load_state_dict(torch.load(path))

intent = classify_intent([sentence], model, max_seq_len=50)

print("Predicted intent = ",intent)

