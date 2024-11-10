from dotenv import dotenv_values

from groq import Groq

from backend.monitor_prompt import unsafe_categories
import logging

config = dotenv_values(".env")

client = Groq(api_key=config["GROQ_API_KEY"])

def moderate_message(message: str):
    """
    Evaluates the user input or AI assistant messages using the Llama Guard model hosted by Groq and handles the response.

    Args:
        message (str): The user or AI assistant message to be evaluated.
    
    Returns:
        str: A warning if the input is unsafe, otherwise a confirmation that the message is safe.
    """
    logging.info("Evaluating message: %s", message)
    
    GUARDRAIL_MODEL: str = "llama-guard-3-8b"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                 "role": "user", 
                 "content": message
                },
            ],
            model=GUARDRAIL_MODEL
        )
        
        response = chat_completion.choices[0].message.content
        logging.info("Model response: %s", response)
        
        if response != 'safe':
            warning_message = "Warning: Your message does not comply with our application rules and responsibilities."
            logging.warning(warning_message)
            return warning_message
        else:
            return "safe"
    
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return "An error occurred while evaluating the message."
