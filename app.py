from flask import Flask, request, jsonify, render_template
import pandas as pd
from chat import get_bot_response  # Import the get_bot_response function from chat.py
import re

app = Flask(__name__)

# Sample medical data (symptoms, causes, medicines)
df_medical = pd.read_csv('medical data.csv')
medical_data = {
    'symptoms': df_medical['Symptoms'].tolist(),
    'causes': df_medical['Causes'].tolist(),
    'diseases': df_medical['Disease'].tolist(),
    'medicine': df_medical['Medicine'].tolist(),
}

# Implement the extract_cause and extract_medicine functions
# Implement the extract_cause and extract_medicine functions
def extract_cause(response):
    # Use regular expressions to find text between "Cause:" and "Medicine:"
    match = re.search(r'cause:(.*?)medicine:', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "Cause not found"

def extract_medicine(response):
    # Use regular expressions to find text after "Medicine:"
    match = re.search(r'medicine:(.*?)$', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "Medicine not found"

    # Use regular expressions to find text within square brackets []
    matches = re.findall(r'\[([^\]]+)\]', response)
    if matches and len(matches) > 1:
        return matches[1]  # Assuming the second match is the medicine
    else:
        return "Medicine not found"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message")
    bot_responses = get_bot_response(user_message)  # Use chat_with_bot to get the bot's response

    # Initialize cause and medicine variables
    cause = "Not found"
    medicine = "Not found"

    # Iterate through bot responses to find cause and medicine
    for response in bot_responses:
      if "cause:" in response:
        cause_match = re.search(r'cause:(.*)', response, re.IGNORECASE)
        if cause_match:
            cause = cause_match.group(1).strip()
            if '.' in cause:
                cause = cause.split('.')[0]  # Extract text before the first dot

      if "medicine:" in response:
        medicine_match = re.search(r'medicine:(.*)', response, re.IGNORECASE)
        if medicine_match:
            medicine = medicine_match.group(1).strip()
            if '.' in medicine:
                medicine = medicine.split('.')[0]  # Extract text before the first dot


    return jsonify({"response": bot_responses, "cause": cause, "medicine": medicine})
if __name__ == "__main__":
    app.run(debug=True)
