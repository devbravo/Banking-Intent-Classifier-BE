import logging
from src.db.session import get_supabase_client
from src.db.models import insert_user_query, insert_feedback
from datetime import datetime, timezone

logging.basicConfig(lefel=logging.INFO)
logger = logging.getLogger(__name__)


def log_query_to_db(query_text: str, predicted_intent: str, 
                    confidence_score: float = None) -> int:
    """
    Log the user query and model prediction to the Supabase database.
    
    Args:
        query_text (str): The user's query text.
        predicted_intent (str): The predicted intent label.
        confidence_score (float, optional): Prediction confidence score.
    """
    supabase = get_supabase_client()
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        result = insert_user_query(supabase, query_text, predicted_intent,
                                   confidence_score, created_at)
        logger.info(f'Logged to datavase: {result}')
        return result[0]['id']
      
    except (KeyError, IndexError) as e:
        logger.error(f"Error accessing result ID: {e}")
        raise
      
    except Exception as e:
        logger.error(f"Error logging to database: {e}")
        raise
      
      
def log_feedback_to_db(query_id: int, is_correct: bool, 
                       corrected_intent: str = None) -> None:
    """
    Log feedback about the prediction to the Supabase database.
    
    Args:
        query_id (int): The ID of the user query being referenced.
        is_correct (bool): Whether the prediction was correct.
        corrected_intent (str, optional): The corrected intent if the 
                                          prediction was incorrect.
    """
    supabase = get_supabase_client()
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        result = insert_feedback(supabase, query_id, is_correct, 
                                 corrected_intent, created_at)
        
        logger.info(f"Feedback logged to database: {result}")
    except Exception as e:
        logger.error(f"Error logging feedback to database: {e}")
        raise