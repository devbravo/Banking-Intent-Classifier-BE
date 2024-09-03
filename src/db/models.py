from supabase import Client


def insert_user_query(supabase: Client, query_text: str, predicted_intent: str,
                      confidence_score: float, timestamp: str):
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
        "query_text": query_text,
        "predicted_intent": predicted_intent,
        "confidence_score": confidence_score,
        "created_at": timestamp,
    }
    
    response = supabase.table("user_queries").insert(data).execute()      
    return response.data