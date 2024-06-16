import pandas as pd
import numpy as np
import torch
from models import DepressiiNet
from torch.utils.data import DataLoader
from datasets import SaDataset
from augs import test_transform, transform
from tqdm import tqdm
import cv2

if __name__=="__main__":
    test_dataset = SaDataset(
            "data_friends.csv", "data_friends.csv", test_transform, train=True
        )

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=0)

    running_acc = 0
    MODEL_PATH  = "model.pth"
    model = DepressiiNet()
    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt)
    dct = {value: key for key,value in test_dataset.class_to_idx.items()}
    model.eval()
    upload_path = "result"
    for i, (x, y) in tqdm(enumerate(test_dataloader)):

        with torch.set_grad_enabled(False):
            preds = model(x)
            preds_class = preds.argmax(dim=1)
            running_acc += (preds_class == y.data).float().mean()
            
            vis = x[0].permute(2, 1, 0).cpu().detach().numpy()
            vis = (vis*255).astype(int)

            cv2.imwrite(f"result/{dct[preds_class.cpu().detach().numpy()[0]]}-{str(i)}.png", vis)
