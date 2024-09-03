from src.db.session import get_supabase_client
from src.db.models import insert_user_query
from datetime import datetime, timezone


def log_query_to_db(query_text: str, predicted_intent: str, 
                    confidence_score: float = None):
    """
    Log the user query and model prediction to the Supabase database.
    
    Args:
        query_text (str): The user's query text.
        predicted_intent (str): The predicted intent label.
        confidence_score (float, optional): Prediction confidence score.
    """
    supabase = get_supabase_client()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        result = insert_user_query(supabase, query_text, predicted_intent,
                                   confidence_score, timestamp)
        print(f"Logged to database: {result}")
    except Exception as e:
        print(f"Error logging to database: {e}")
        raise