from keyPointsDatasets.keyPointsDatasets import KeyPointDatasets
from torch.utils.data import dataloader,dataset
from torchvision import models
import torch

train_dataset = KeyPointDatasets("tennis_court_dataset/data/images","tennis_court_dataset/data/data_train.json")
val_dataset = KeyPointDatasets("tennis_court_dataset/data/images","tennis_court_dataset/data/data_val.json")

train_loader = dataloader(train_dataset, batch_size = 8, shuffle =True)
val_loader = dataloader(val_dataset, batch_size = 8, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained = True)
model.fc = torch.nn.Linear(model.fc.in_features,14*2)
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20
for epoch in range(epochs) :
    for i, (img , kps) in enumerate(train_loader) :
        img = img.to(device)
        kps = kps.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs,kps)
        loss.backward()
        optimizer.step()

        if i% 10 == 0:
            print(f"Epoch {epoch},iter {i},loss: {loss.item()}")

torch.save(model.state_dict(),"keypoints_model.pth")