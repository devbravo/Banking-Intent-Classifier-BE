import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


def get_supabase_client() -> Client:
    """
    Returns the Supabase client for interacting with the database.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and API key must be set in the env")
    
    supabase: Client = create_client(supabase_url,
                                     supabase_key)
    return supabase