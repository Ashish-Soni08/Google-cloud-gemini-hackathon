from dotenv import dotenv_values

from groq import Groq

from backend.monitor_prompt import unsafe_categories

config = dotenv_values(".env")

client = Groq(api_key=config["GROQ_API_KEY"])

def evaluate_input(user_message: str):
    """
    Evaluates the user input using the Llama Guard model hosted by Groq.

    Args:
        user_message (str): The user input to be evaluated.
    
    Returns:
        str: The response from the model indicating if the input is safe or unsafe.
    """
    # MODEL HOSTED BY GROQ
    GUARDRAIL_MODEL: str = "llama-guard-3-8b"
    chat_completion = client.chat.completions.create(
        messages=[
            {
             "role": "user", 
             "content": user_message
            },
        ],
        model=GUARDRAIL_MODEL # llama-guard-3-8b
    )

    response = chat_completion.choices[0].message.content
    return response

def do_something_with_response(response: str):
    """
    Placeholder function to demonstrate how to use the response from the model.

    Args:
        response (str): The response from the model.
    """
    if response in unsafe_categories:
        print("The user input is unsafe.")
    else:
        print("The user input is safe.")
