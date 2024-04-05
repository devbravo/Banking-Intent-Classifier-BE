from pathlib import Path 
import torch 
import bentoml 
import yaml
from intent_classifier import IntentClassifier
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir,  '../../config.yaml')

with open(file_path, 'r') as f: 
  config = yaml.safe_load(f)

device = 'mps' if torch.cuda.is_available() else 'cpu'

def load_model_and_save_to_bento(model_file: Path) -> None: 
  model = IntentClassifier(config=config)
  print(model.embedding)
  model.to(device)
  model.load_state_dict(torch.load(model_file, map_location=device))
  bento_moodel = bentoml.pytorch.save_model('pytorch-intent-classifier', model)
  print(f'Bento model tag = {bento_moodel.tag}')
  
if __name__ == '__main__':
  current_dir = os.path.dirname(os.path.realpath(__file__))
  model_path = os.path.join(current_dir,  'model.pth')
  load_model_and_save_to_bento(Path(model_path))