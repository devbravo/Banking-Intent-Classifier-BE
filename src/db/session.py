from supabase import create_client, Client
import yaml
import os


def load_db_config():
    """
    Load the database configuration from a YAML file.
    """
    config_path = os.path.join(os.path.dirname(__file__),
                               '../../config/db_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
           

db_config = load_db_config()

supabase: Client = create_client(db_config['supabase_url'],
                                 db_config['supabase_key'])


def get_supabase_client() -> Client:
    """
    Returns the Supabase client for interacting with the database.
    """
    return supabase