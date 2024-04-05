from pathlib import Path 
import torch 
import bentoml 
import yaml
import os
import sys
from classifier_model import IntentClassifier

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir,  './config.yaml')

with open(file_path, 'r') as f: 
  config = yaml.safe_load(f)

device = 'mps' if torch.cuda.is_available() else 'cpu'

def load_model_and_save_to_bento(model_file: Path) -> None: 
  lstm_model = IntentClassifier(config).to(device)
  lstm_model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
  bento_model = bentoml.pytorch.save_model('classifier', model=lstm_model)
  print(f'Bento model tag = {bento_model.tag}')
  
if __name__ == '__main__':
  current_dir = os.path.dirname(os.path.realpath(__file__))
  model_path = os.path.join(current_dir,  './class_model.pth')
  print('path', model_path)
  
  load_model_and_save_to_bento(Path(model_path))