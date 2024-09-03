from supabase import Client
from uuid import uuid4
import datetime


def insert_user_query(supabase: Client, query_text: str, predicted_intent: str,
                      confidence_score: float = None, user_id=None):
    """
    Insert a new user query into the database.
    Args:
        supabase (Client): The Supabase client instance.
        query_text (str): The text of the user query.
        predicted_intent (str): The predicted intent label.
        confidence_score (float, optional): Prediction confidence score.
        user_id (str, optional): The ID of the user (if available).
    
    Returns:
        dict: The inserted record.
    """
    data = {
        "id": str(uuid4()),
        "query_text": query_text,
        "predicted_intent": predicted_intent,
        "confidence_score": confidence_score,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "user_id": user_id
    }
    
    response = supabase.table("user_queries").insert(data).execute()
    return response.data