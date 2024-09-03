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
import torch.nn.functional as F
import bentoml
from bentoml.io import Text, JSON
from src.utils.label_mapping import label_mapping
from src.data_preprocessing.text_processing import clean_text, lemmatizer
from src.data_preprocessing.text_processing import numericalize
from src.utils.get_device import get_device
from src.api.database import log_query_to_db, log_feedback_to_db


DEVICE = get_device()
print(f"Using device: {DEVICE}")

with open('data/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

classifier = bentoml.pytorch.get('classifier:latest').to_runner()
svc = bentoml.Service('classifier', runners=[classifier])


@svc.api(input=Text(), output=JSON())
def inference(text: str) -> dict:
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
        logits = classifier.run(tensor_text)
        probas = F.softmax(logits, dim=1).cpu().numpy()
        pred_index = torch.argmax(logits, dim=1).cpu().numpy()[0]
        confidence_score = float(probas[0][pred_index])
        predicted_intent = label_mapping[pred_index]
        
        query_id = log_query_to_db(text, predicted_intent, confidence_score)
        # print("Query_id", query_id)
        
        print(f"Predicted_labe:, {predicted_intent}", 
              f"confidence_score: {confidence_score}")
        
        return {
            "predicted_intent": predicted_intent,
            "confidence_score": confidence_score,
            "query_id": query_id
        }
      

@svc.api(input=JSON(), output=JSON())
def submit_feedback(feedback_data: dict) -> dict:
    """
    Submit feedback about the prediction.

    Args:
        feedback_data (dict): A dictionary containing feedback information. 
        Expected keys:
            - query_id (int): The ID of the user query.
            - is_correct (bool): Whether the prediction was correct.
            - corrected_intent (str, optional): The corrected intent if the 
                                                prediction was incorrect.
    Returns:
        dict: A confirmation message.
    """
    query_id = feedback_data.get("query_id")
    is_correct = feedback_data.get("is_correct")
    corrected_intent = feedback_data.get("corrected_intent", None)

    if not query_id:
        return {"error": "Query ID is missing."}, 400
    if is_correct is None:
        return {"error": "is_correct field is required"}, 400

    try:
        log_feedback_to_db(query_id, is_correct, corrected_intent)
        return {"message": "Feedback submitted successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500