import torch 
import yaml
from src.model.intent_classifier import IntentClassifier

with open('config.yaml', 'r') as f: 
  config = yaml.safe_load(f)
  
model = IntentClassifier(config=config)