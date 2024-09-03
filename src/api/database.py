from src.db.session import get_supabase_client
from src.db.models import insert_user_query


def log_query_to_db(query_text: str, predicted_intent: str, 
                    confidence_score: float = None, user_id=None):
    """
    Log the user query and model prediction to the Supabase database
    Args:
        query_text (str): The user's query text.
        predicted_intent (str): The predicted intent label.
        confidence_score (float, optional): Prediction confidence score.
        user_id (str, optional): The ID of the user (if available).
    """
    supabase = get_supabase_client()
    result = insert_user_query(supabase, query_text, predicted_intent,
                               confidence_score, user_id)
    print(f"Logged to database: {result}")