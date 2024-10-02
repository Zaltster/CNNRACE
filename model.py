import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class FaceDataset(Dataset):
    def __init__(self, X, y_age, y_ethnicity, y_gender):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        self.y_age = torch.FloatTensor(y_age)
        self.y_ethnicity = torch.LongTensor(y_ethnicity)
        self.y_gender = torch.LongTensor(y_gender)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_age[idx], self.y_ethnicity[idx], self.y_gender[idx]

class FaceCNN(nn.Module):
    def __init__(self, num_ethnicities, num_genders):
        super(FaceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        self.age = nn.Sequential(
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        self.ethnicity = nn.Sequential(
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_ethnicities)
        )
        
        self.gender = nn.Sequential(
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_genders)
        )
    
    def forward(self, x):
        x = self.features(x)
        age = self.age(x)
        ethnicity = self.ethnicity(x)
        gender = self.gender(x)
        return age, ethnicity, gender

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    
    X = np.array([np.array(pixels.split()) for pixels in df['pixels']], dtype='float32')
    X = X.reshape(-1, 48, 48) / 255.0  # Normalize pixel values
    
    le_ethnicity = LabelEncoder()
    le_gender = LabelEncoder()
    
    y_age = df['age'].values
    y_ethnicity = le_ethnicity.fit_transform(df['ethnicity'])
    y_gender = le_gender.fit_transform(df['gender'])
    
    return X, y_age, y_ethnicity, y_gender

def train_model(model, train_loader, val_loader, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion_age = nn.MSELoss()
    criterion_ethnicity = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, ages, ethnicities, genders = [b.to(device) for b in batch]
            optimizer.zero_grad()
            age_pred, ethnicity_pred, gender_pred = model(inputs)
            loss_age = criterion_age(age_pred.squeeze(), ages)
            loss_ethnicity = criterion_ethnicity(ethnicity_pred, ethnicities)
            loss_gender = criterion_gender(gender_pred, genders)
            loss = loss_age + 0.5 * loss_ethnicity + 0.5 * loss_gender
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, ages, ethnicities, genders = [b.to(device) for b in batch]
                age_pred, ethnicity_pred, gender_pred = model(inputs)
                loss_age = criterion_age(age_pred.squeeze(), ages)
                loss_ethnicity = criterion_ethnicity(ethnicity_pred, ethnicities)
                loss_gender = criterion_gender(gender_pred, genders)
                loss = loss_age + 0.5 * loss_ethnicity + 0.5 * loss_gender
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")

def main():
    csv_file = 'age_gender.csv'
    X, y_age, y_ethnicity, y_gender = load_data(csv_file)
    
    X_train, X_test, y_train_age, y_test_age, y_train_ethnicity, y_test_ethnicity, y_train_gender, y_test_gender = train_test_split(
        X, y_age, y_ethnicity, y_gender, test_size=0.2, random_state=42)
    
    train_dataset = FaceDataset(X_train, y_train_age, y_train_ethnicity, y_train_gender)
    test_dataset = FaceDataset(X_test, y_test_age, y_test_ethnicity, y_test_gender)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    num_ethnicities = len(np.unique(y_ethnicity))
    num_genders = len(np.unique(y_gender))
    model = FaceCNN(num_ethnicities, num_genders)
    
    train_model(model, train_loader, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'face_cnn_model.pth')

if __name__ == "__main__":
    main()