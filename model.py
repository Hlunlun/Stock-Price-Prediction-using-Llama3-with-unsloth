
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_shape):
        super(StockLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape[1], hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 1)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc1(x[:, -1, :])  # Taking the last time step output
        x = self.fc2(x)
        return x


class StockLLM:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_path)
        self.device = args.device
    def predict(self):
        pass

    def load_model(self):
        pass



    def generate(self, prompt, data=None):

        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(self.device)
        outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1, temperature=0.7)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text