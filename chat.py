import random
import pandas as pd
import torch
import json

from model.model import NeuralNet
from nltk_utils.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the medical data CSV file
df = pd.read_csv('medical data.csv')

# Extract relevant columns
symptoms = df['Symptoms'].tolist()
causes = df['Causes'].tolist()
diseases = df['Disease'].tolist()
medicine = df['Medicine'].tolist()

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SAAD"

def get_bot_response(sentence):
    bot_response = []  # Store responses in a list

    # Add your code here to match user input with medical data
    found_match = False
    for i in range(len(symptoms)):
        symptom_str = str(symptoms[i])  # Ensure the value is treated as a string
        if sentence.lower() in symptom_str.lower():
            response = f"{bot_name}: You may be experiencing {diseases[i]}. You can take medicine:{medicine[i]}.cause:{causes[i]}"
            bot_response.append(response)
            found_match = True
            break
    
    if not found_match:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = f"{bot_name}: {random.choice(intent['responses'])}"
                    bot_response.append(response)
        else:
            response = f"{bot_name}: I do not understand..."
            bot_response.append(response)
    
    return bot_response

    bot_response = []  # Store responses in a list

    # Add your code here to match user input with medical data
    found_match = False
    for i in range(len(symptoms)):
        if sentence.lower() in symptoms[i].lower():
            response = f"{bot_name}: You may be experiencing {diseases[i]}. You can take {medicine[i]} for relief."
            bot_response.append(response)
            found_match = True
            break
    
    if not found_match:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = f"{bot_name}: {random.choice(intent['responses'])}"
                    bot_response.append(response)
        else:
            response = f"{bot_name}: I do not understand..."
            bot_response.append(response)
    
    return bot_response
