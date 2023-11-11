from hand_pose_model import HandPoseModel
from map_img_to_hand_data import HandPoseDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# print(HandPoseModel())

def train_one_epoch(train_loader, epoch_index, tb_writer, optim, scheduler):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        img, hp = data
        img, hp = img.to(device), hp.to(device)

        optim.zero_grad()

        pred_hp = model(img).reshape(21,3)

        loss = loss_fn(pred_hp, hp)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if i % 1000 === 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    
    return last_loss

if __name__ == "__main__":
 
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) ])

    hp_dataset = HandPoseDataset(transform=transform)
    print(len(hp_dataset))

    train_data, val_data, test_data = torch.utils.data.random_split(hp_dataset, [0.85, 0.05, 0.10])
    print(len(train_data), len(val_data), len(test_data))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:",device)

    model = nn.DataParallel(HandPoseModel()).to(device) # HandPoseModel()
    sample = train_data[0][0].unsqueeze(0).to(device)
    print(model(sample))
    # print(sample.shape)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"runs/hand_pose_trainer_{timestamp}")

    epoch_number = 0

    EPOCHS = 10

    best_vloss = float("inf")

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch_number + 1}:")

        model.train(True)
        avg_loss = train_one_epoch(train_loader, epoch_number, writer, optimizer, scheduler)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i+1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'hand_pose_model/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
  