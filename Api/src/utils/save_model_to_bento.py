from pathlib import Path 
import torch 
import bentoml 
import yaml
from src.models.intent_classifier import IntentClassifier

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

with open('../models/model_config.yaml', 'r') as f: 
  config = yaml.safe_load(f)

def load_model_and_save_to_bento(model_file: Path) -> None: 
  """
    Load a trained PyTorch model from a file and save it to BentoML.

    Args:
        model_file (Path): Path to the trained PyTorch model's state dictionary.
    """
  lstm_model = IntentClassifier(config).to(device)
  lstm_model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
  bento_model = bentoml.pytorch.save_model('classifier', model=lstm_model)
  print(f'Bento model tag = {bento_model.tag}')
  
if __name__ == '__main__':
  model_path = '/model/class_model.pth'
  load_model_and_save_to_bento(Path(model_path))