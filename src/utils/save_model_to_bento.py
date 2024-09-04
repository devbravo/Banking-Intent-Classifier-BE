"""
This module is responsible for loading a pre-trained PyTorch model,
configuring it according to the specified device (CUDA, MPS, or CPU),
and saving it to BentoML for deployment.
The model configuration is loaded from a YAML file, and the module
also provides functionality to save the model using BentoML, which facilitates
model deployment and management in production environments.

The module includes:
- Device selection logic to determine whether to use CUDA, MPS, or CPU.
- Loading of model configurations from a YAML file.
- A function to load the PyTorch model from a specified file and save it.
- An entry point for executing model loading & saving process.

Dependencies:
- PyTorch is used for loading and managing the model.
- BentoML is used for saving and deploying the model.
- YAML is used for configuration management.
"""

from pathlib import Path
import torch
import bentoml
import yaml
from src.models.intent_classifier import IntentClassifier
from utils.get_device import get_device

DEVICE = get_device()
print(f"Using device: {DEVICE}")

with open('../models/model_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def load_model_and_save_to_bento(model_file: Path) -> None:
    """
      Load a trained PyTorch model from a file and save it to BentoML.
      Args:
          model_file (Path): Path to the trained PyTorch model.
      Raises:
        FileNotFoundError: If the model file is not found at the specified 
                           path.
        RuntimeError: If there is an issue loading the model or saving it to 
                          BentoML.
      """
    try:
        lstm_model = IntentClassifier(config).to(DEVICE)
        lstm_model.load_state_dict(torch.load(
            model_file,
            map_location=torch.device(DEVICE),
            weights_only=True
        ))
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        raise RuntimeError(f"Failed to load model from {model_file}: {e}")
      
    try:
        bento_model = bentoml.pytorch.save_model('classifier',
                                                 model=lstm_model)
        print(f'Bento model tag = {bento_model.tag}')
    except Exception as e:
        raise RuntimeError(f"Failed to save Bento model: {e}")
    
      
if __name__ == '__main__':
    MODEL_PATH = '/model/class_model.pth'
    load_model_and_save_to_bento(Path(MODEL_PATH))