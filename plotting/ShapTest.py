import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# generate synthetic data

X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)

print(f"Training data shape: {X_train_tensor.shape}")
print(f"Testing data shape:  {X_test_tensor.shape}\n")

# build the neural network
class SimpleBINN(nn.Module):
    def __init__(self):
        super(SimpleBINN, self).__init__()
        self.layer1 = nn.Linear(20, 10)  # 20 input genes to hidden layer
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)   # Hidden layer to output
        self.sigmoid = nn.Sigmoid()      # Binary classification (0 or 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = SimpleBINN()

# train the model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training model...")
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Training complete.\n")

# deepshap explainer
model.eval()

# Create the background dataset 
background = X_train_tensor

# Select target patients to explain
test_patients = X_test_tensor[:50] # explaining the first 50 test patients

print("Initializing DeepExplainer and calculating SHAP values...")
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_patients)

print(f"SHAP values: {shap_values}")

# inspect the format
print("\nformat inspection:")

if isinstance(shap_values, list):
    print("SHAP returned a list (one array per class). Taking the positive class.")
    shap_matrix = shap_values[1]
else:
    print("SHAP returned a single numpy array.")
    shap_matrix = shap_values


shap_matrix = np.squeeze(shap_matrix) 

print(f"Shape of the SHAP Matrix: {shap_matrix.shape} -> (Patients, Genes)")

patient_0_shaps = shap_matrix[0]
patient_0_genes = test_patients[0].numpy()

gene_names = [f"Gene_{i}" for i in range(20)]

df = pd.DataFrame({
    'Gene_Name': gene_names,
    'Raw_Gene_Expression': patient_0_genes,
    'SHAP_Impact': patient_0_shaps  
})

# Sort by absolute impact to find the top drivers
df['Absolute_Impact'] = df['SHAP_Impact'].abs()
df_sorted = df.sort_values(by='Absolute_Impact', ascending=False).drop(columns=['Absolute_Impact'])

print("\nTop 5 Driving Genes for Patient 0:")
print(df_sorted.head())

print("\nGENERATING PLOTS...")

# Extract the baseline expected value
base_value = explainer.expected_value


if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
    base_value = base_value[0]
if torch.is_tensor(base_value):
    base_value = base_value.item()

# Build the SHAP explanation object manually
shap_explanation = shap.Explanation(
    values=shap_matrix,                  
    base_values=base_value,              
    data=test_patients.numpy(),          
    feature_names=gene_names  
)

# generate the plots
print("Displaying Beeswarm Plot...")
shap.plots.beeswarm(shap_explanation)
plt.show()  

print("Displaying Waterfall Plot for Patient 0...")
shap.plots.waterfall(shap_explanation[0])

print("Displaying Violin Plot...")
shap.plots.violin(shap_explanation)

plt.show()
