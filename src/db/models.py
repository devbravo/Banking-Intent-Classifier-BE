from supabase import Client


def insert_user_query(supabase: Client, query_text: str, predicted_intent: str,
                      confidence_score: float, created_at: str):
    """
    Insert a new user query into the database.
    Args:
        supabase (Client): The Supabase client instance.
        query_text (str): The text of the user query.
        predicted_intent (str): The predicted intent label.
        confidence_score (float, optional): Prediction confidence score.
        created_at (str, optional): The timestamp when the query was \
                                   submitted.
    Returns:
        dict: The inserted record.
     Raises:
        Exception: If the insert operation fails.
    """
    data = {
        "query_text": query_text,
        "predicted_intent": predicted_intent,
        "confidence_score": confidence_score,
        "created_at": created_at,
    }
    
    try:
        response = supabase.table("user_queries").insert(data).execute()      
        if response.status_code != 201:
            status_code = response.status_code
            error_details = response.model_dump_json()
            raise Exception(
              f"Failed to insert user query: {status_code} - {error_details}")
        return response.data
  
    except Exception as e:
        print(f"Error inserting user query: {e}")
        raise
  
  
def insert_feedback(supabase: Client, query_id: int, is_correct: bool,
                    corrected_intent: str = None, created_at: str = None):
    """
    Insert feedback into the feedback table.
    Args:
        supabase (Client): The Supabase client instance.
        query_id (int): The ID of the query in the user_queries table.
        is_correct (bool): Whether the prediction was correct.
        corrected_intent (str, optional): Corrected intent if incorrect \
                                          prediction.
        created_at (str, optional): The timestamp when the feedback was \
                                   submitted.
    Returns:
        dict: The inserted feedback record.
    Raises:
        Exception: If the insert operation fails.
    """
    
    data = {
        "query_id": query_id,
        "is_correct": is_correct,
        "corrected_intent": corrected_intent,
        "created_at": created_at
    }
    try:
        response = supabase.table("feedback").insert(data).execute()
        if response.status_code != 201:
            status_code = response.status_code
            error_details = response.model_dump_json()
            raise Exception(
              f"Failed to insert feedback: {status_code} - {error_details}")
        return response.data

    except Exception as e:
        print(f"Error inserting feedback: {e}")
        raise