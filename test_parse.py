import re
from dash import html

def parse_summary(summary_text):
    """Parse the AI-generated summary into structured Q&A format for patient clarity."""
    print("[Postprocess] Parsing the summary into Q&A format...")

    # Initialize a dictionary to store the parsed data
    qna = {
        "use": "",
        "dosage": "",
        "side_effects": "",
        "precautions": ""
    }

    # Split the summary by lines
    lines = summary_text.split("\n")
    
    # Prepare placeholders for questions and answers
    current_question = None
    current_answer = []

    # Define the phrases to check for
    phrase_to_check = {
        "use": "what is this medicine for?",
        "dosage": "how is it used?",
        "side_effects": "what are the common side effects?",
        "precautions": "are there any important precautions or warnings for patients?"
    }

    for line in lines:
        line = line.strip()

        # Check for question phrases (exact matching)
        for question, phrase in phrase_to_check.items():
            if phrase.lower() in line.lower():  # Check if the phrase exists in the line
                if current_question and current_answer:
                    qna[current_question] = " ".join(current_answer).strip()
                current_question = question
                current_answer = []  # Reset for the new question
                break  # Stop checking after the first match

        # If line contains the answer, extract the text after ": "
        if '- Answer:' in line:
            # Split by ': ' and get the part after it, strip any extra whitespace
            answer = line.split(': ', 1)[1].strip()
            current_answer.append(answer)

    # Ensure the last question's answer is also added
    if current_question and current_answer:
        qna[current_question] = " ".join(current_answer).strip()

    # Return the structured data for Q&A, formatted into HTML list items
    return [
        html.Li(f"Use: {qna['use']}"),
        html.Li(f"Dosage: {qna['dosage']}"),
        html.Li(f"Common Side Effects: {qna['side_effects']}"),
        html.Li(f"Precautions: {qna['precautions']}")
    ]

# Sample summary text
summary_text = """ ---
    - What is this medicine for?
    - Answer: Molnupiravir is a drug used to prevent severe outcomes such as hospitalization and death due to COVID-19 in adults at increased risk of severe disease.
    
    - How is it used?
    - Answer: Molnupiravir is taken as a single 800mg dose, followed by 400mg twice daily for 5 days. It should be taken with food.
    
    - What are the common side effects?
    - Answer: Common side effects include nausea, diarrhea, and headache.
    
    - Are there any important precautions or warnings for patients?
    - Answer: Molnupiravir should not be used by people with severe kidney or liver disease. Pregnant or breastfeeding women should not take molnupiravir. People with a history of alcoholism should use molnupiravir with caution as it may increase the risk of liver damage. Molnupiravir may interact with other medications, so patients should inform their healthcare provider of all medications they are taking.
[Postprocess] Parsing the summary into Q&A format..."""

# Testing the function locally
parsed_summary = parse_summary(summary_text)
print(parsed_summary)
