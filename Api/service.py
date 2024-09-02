import os
import torch
import bentoml 
import json
from bentoml.io import Text
from src.utils.label_mapping import label_mapping
from src.data_preprocessing.text_processing import clean_text, lemmatizer, numericalize
from src.models.intent_classifier import *

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")

with open('data/vocab.json', 'r') as f:
    vocab = json.load(f)
    
classifier = bentoml.pytorch.get('classifier:latest').to_runner()
svc = bentoml.Service('classifier', runners=[classifier])

@svc.api(input=Text(), output=Text()) 
def inference(text: str) -> str:
  """
    Perform inference on input text to predict the customer's intent.
    Args:
        text (str): The input text from the user for which the intent is to be predicted.
    Returns:
        str: The predicted label representing the customer's intent.
    """
  cleaned_text = clean_text(text)
  lemmatized_text = lemmatizer(cleaned_text)
  numericalized_text = numericalize(vocab, lemmatized_text)
  tensor_text = torch.tensor(numericalized_text).to(device)
  with torch.no_grad():
    pred = classifier.run(tensor_text)
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    predicted_label = label_mapping[pred[0]]
    print("Predicted_label", predicted_label)

    return predicted_label