from flask import Flask, request, jsonify
import torch
import torch.nn as nn

# Define your model
class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x

app = Flask(__name__)

# Load the trained model
model = StudentModel(input_dim=59)  # Adjust input_dim based on your model
model.load_state_dict(torch.load("best_student_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = torch.tensor(data['input'], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)
    prediction = output.item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
