import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Model):
    def __init__(self,input_dim):
          super().__init__()
          self.fc1 = nn.Linear(input_dim,30)
          self.fc2 = nn.Linear(30,15)
          self.fc3 = nn.Linear(15,1)
          self.activate = torch.relu

    def forward(self,x):
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation for regression)
        return x

if __name__ == "__main__":
    input_dim = 50
    lr = 0.001
    num_epochs = 500

    model = SimpleNet(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr = lr)
    criterion = nn.MSELoss()

    x_train = torch.rand(1000, input_dim)
    y_train = 7 * x_train.sum(dim=1, keepdim=True)

    for epoch in range(num_epochs):
        model.train()

        output = model(x_train)
        loss = criterion(output,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Save all model
    torch.save(model, "model.pth")
    
    #Save weights only
    torch.save(model.state_dict(), 'model.pth')

    model.eval()
    with torch.no_grad():
        for x in range(7):
            x_test = torch.rand(1, input_dim)
            y_pred = model(x_test)
            y_true = 7 * x_test.sum(dim=1, keepdim=True)
            print(f"Predicted: {y_pred.item():0.2f} | Ground Truth: {y_true.item():0.2f}")

