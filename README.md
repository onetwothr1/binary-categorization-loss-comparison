# binary-categorization-loss-comparison


An experiment with toy problem.

  ```.py
  import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Generate toy data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

# Convert data to tensors
x_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y)

# Define the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Initialize the model
model1 = Model()
model2 = Model()
model3 = Model()

# Define the loss functions
BCE = nn.BCELoss()
L1 = nn.L1Loss()
L2 = nn.MSELoss()

# Define the optimizer
optimizer1 = optim.SGD(model1.parameters(), lr=0.05)
optimizer2 = optim.SGD(model2.parameters(), lr=0.05)
optimizer3 = optim.SGD(model3.parameters(), lr=0.05)

# Training loop
num_epochs = 40000
bce_losses = []
l1_losses = []
l2_losses = []
y0_pred_avg_bce = []
y0_pred_avg_l1 = []
y0_pred_avg_l2 = []
y1_pred_avg_bce = []
y1_pred_avg_l1 = []
y1_pred_avg_l2 = []

for epoch in tqdm(range(num_epochs)):
    # Forward pass
    bce_output = model1(x_tensor)
    bce_loss = BCE(bce_output, y_tensor)
    l1_output = model2(x_tensor)
    l1_loss = L1(l1_output, y_tensor)
    l2_output = model3(x_tensor)
    l2_loss = L2(l2_output, y_tensor)

    # Backward and optimize
    optimizer1.zero_grad()
    bce_loss.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    l1_loss.backward()
    optimizer2.step()

    optimizer3.zero_grad()
    l2_loss.backward()
    optimizer3.step()

    # Store the losses
    bce_losses.append(bce_loss.item())
    l1_losses.append(l1_loss.item())
    l2_losses.append(l2_loss.item())

    # Calculate average predicted value for y = 0 and 1
    y0_indices = torch.where(y_tensor == 0)[0]
    y1_indices = torch.where(y_tensor == 1)[0]

    y0_pred_avg_bce.append(torch.sum(bce_output[y0_indices]).detach().numpy() / y0_indices.size(0))
    y0_pred_avg_l1.append(torch.sum(l1_output[y0_indices]).detach().numpy() / y0_indices.size(0))
    y0_pred_avg_l2.append(torch.sum(l2_output[y0_indices]).detach().numpy() / y0_indices.size(0))
    y1_pred_avg_bce.append(torch.sum(bce_output[y1_indices]).detach().numpy() / y1_indices.size(0))
    y1_pred_avg_l1.append(torch.sum(l1_output[y1_indices]).detach().numpy() / y1_indices.size(0))
    y1_pred_avg_l2.append(torch.sum(l2_output[y1_indices]).detach().numpy() / y1_indices.size(0))

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the loss on the left subplot
ax1.plot(bce_losses, label='BCE Loss')
ax1.plot(l1_losses, label='L1 Loss')
ax1.plot(l2_losses, label='L2 Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Binary Classification Losses')
ax1.legend()

# Plot the average predicted values on the right subplot
ax2.plot(y0_pred_avg_bce, label='y=0 (BCE)')
ax2.plot(y0_pred_avg_l1, label='y=0 (L1)')
ax2.plot(y0_pred_avg_l2, label='y=0 (L2)')
ax2.plot(y1_pred_avg_bce, label='y=1 (BCE)')
ax2.plot(y1_pred_avg_l1, label='y=1 (L1)')
ax2.plot(y1_pred_avg_l2, label='y=1 (L2)')
ax2.set_title('Average Predicted Values')
ax2.set_ylim([0, 1])
ax2.legend()
plt.tight_layout()
plt.show()
  ```
  
  ![graph](https://github.com/onetwothr1/binary-categorization-loss-comparison/assets/83393021/1d15f93b-cd7d-4e99-b604-cd59f62e4ca9)
