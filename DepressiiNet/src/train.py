import torch
# from loguru import logger
from torch.utils.data import DataLoader

from datasets import SaDataset
from tqdm import tqdm
from models import DepressiiNet
from augs import test_transform, transform
from torchvision import models

def train_model_by_torch(model, dataloader, loss, optimizer, scheduler, experiment_name, n_epochs=10):
    loss_history = []
    acc_history = []

    for epoch in range(n_epochs):
        model.train()

        running_loss = 0
        running_acc = 0

        for x, y in tqdm(dataloader):
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                preds = model(x)
                loss_value = loss(preds, y)
                loss_value.backward()
                preds_class = preds.argmax(dim=1)
                optimizer.step()

            running_loss += loss_value.item()
            running_acc += (preds_class == y.data).float().mean()

        scheduler.step()

        print("EPOCH - ", epoch + 1)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_acc / len(dataloader)

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc.item())
        print("Epoch loss -", epoch_loss)
        print("Epoch accuracy -", epoch_acc.numpy())
        torch.save(model.state_dict(), f"{experiment_name}.pth")


    return model, loss_history, acc_history


if __name__ == "__main__":

    # Инициализируем датасеты
    train_dataset = SaDataset("data_friends.csv", "data_friends.csv", transform)

    # Инициализируем даталодеры
    batch_size = 1
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=0
    )

    # logger.info(f"Length of dataloader{len(train_dataloader)}")

    #model = FriendNet()
    model = models.resnet18(pretrained=True)
    # vacate grads
    for param in model.parameters():
        param.requires_grad = False

    # we have 6 classes
    model.fc = torch.nn.Linear(model.fc.in_features, 6)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # model, loss_his, acc_his = train_model_by_torch(
    #     model, train_dataloader, loss, optimizer, scheduler=scheduler, experiment_name="model", n_epochs=40
    # )

    ###


    model, loss_his, acc_his = train_model_by_torch(
        model, train_dataloader, loss, optimizer, scheduler=scheduler, experiment_name="model_resnet", n_epochs=40
    )
