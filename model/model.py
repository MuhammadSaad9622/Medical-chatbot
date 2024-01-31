import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, medical_data=None):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.medical_data = medical_data

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

    def medical_response(self, input_text):
        if self.medical_data is not None:
            found_match = False
            for i in range(len(self.medical_data['symptoms'])):
                if input_text.lower() in self.medical_data['symptoms'][i].lower():
                    found_match = True
                    response = f"You may be experiencing {self.medical_data['diseases'][i]}. You can take {self.medical_data['medicine'][i]} for relief."
                    break

            if found_match:
                return response

        return "I do not understand..."
