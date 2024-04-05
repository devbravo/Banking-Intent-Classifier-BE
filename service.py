import os
import torch
import bentoml 
import json
from bentoml.io import Text
from src.processing_pipeline.label_mapping import label_mapping
from src.processing_pipeline.text_processing import clean_text, lemmatizer, numericalize
from classifier_model import *

device = 'mps' if torch.cuda.is_available() else 'cpu'
current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, './data/vocab.json')

with open(vocab_path, 'r') as f:
    vocab = json.load(f)
    

classifier = bentoml.pytorch.get('classifier:latest').to_runner()
svc = bentoml.Service('classifier', runners=[classifier])

@svc.api(input=Text(), output=Text()) 
def inference(text: str):
  cleaned_text = clean_text(text)
  lemmatized_text = lemmatizer(cleaned_text)
  numericalized_text = numericalize(vocab, lemmatized_text)
  tensor_text = torch.tensor(numericalized_text)
  with torch.no_grad():
    pred = classifier.run(tensor_text)
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    predicted_label = label_mapping[pred[0]]
    print("Predicted_label", predicted_label)

    return predicted_label