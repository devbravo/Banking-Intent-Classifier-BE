"""
This module sets up and serves an inference API for predicting customer intent
using a pre-trained Bidirectional LSTM model.
The model is managed using BentoML and deployed as a service that takes in
text input and returns the predicted intent label.

The module includes the following components:
- Device selection logic to determine whether to use CUDA, MPS, or CPU.
- Vocabulary loading from a JSON file to support text preprocessing.
- BentoML service definition that wraps the model as an API for inference.

The inference function performs the following steps:
1. Cleans and preprocesses the input text.
2. Converts the processed text into a tensor that the model can use.
3. Runs the model to predict the intent label.
4. Returns the predicted label as a string.
"""

import json
import torch
import bentoml
from bentoml.io import Text
from src.utils.label_mapping import label_mapping
from src.data_preprocessing.text_processing import clean_text, lemmatizer
from src.data_preprocessing.text_processing import numericalize
from src.utils.get_device import get_device


DEVICE = get_device()
print(f"Using device: {DEVICE}")

with open('data/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

classifier = bentoml.pytorch.get('classifier:latest').to_runner()
svc = bentoml.Service('classifier', runners=[classifier])


@svc.api(input=Text(), output=Text())
def inference(text: str) -> str:
    """
      Perform inference on input text to predict the customer's intent.
      Args:
          text (str): Input text from the user for which the intent is to \
                      be predicted.
      Returns:
          str: The predicted label representing the customer's intent.
    """
    cleaned_text = clean_text(text)
    lemmatized_text = lemmatizer(cleaned_text)
    numericalized_text = numericalize(vocab, lemmatized_text)
    tensor_text = torch.tensor(numericalized_text).to(DEVICE)
    with torch.no_grad():
        pred = classifier.run(tensor_text)
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        predicted_label = label_mapping[pred[0]]
        print("Predicted_label", predicted_label)
        return predicted_label