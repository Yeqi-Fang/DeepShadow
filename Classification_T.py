import datetime
import os
import cv2
import torch
import re
import glob
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from math import ceil
from pathlib import Path
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

num_imgaes = 100
height = 1024
width = 1024
shape = 'rect'
BH_lower = 64
BH_upper = 75
wl = 100e-9
D = 6.5
F = 131.4
SIZE = 240
num_epochs = 30
BATCH_SIZE = 80
DROPOUT_RATE = 0.5
learning_rate = 1e-4
weight_decay = 1e-4
critical_acc = 0.4
root_dir = Path('tele_datasets')
b = re.compile(r'AS(\d\.\d*e-\d*)_')

for data_dir in root_dir.glob('reg_num3_rect_wl1.000e-07_D6.50_F131.4_AS1.00e-04_BHSize64-75*'):
    angular_pixel_size_input_image = float(re.findall(b, str(data_dir))[0])
    # if angular_pixel_size_input_image not in [3e-5, 4e-5, 6e-5, 7e-4, 5e-5]:
    #     continue
    print(f'starting ----------------------{angular_pixel_size_input_image:.3e}')
    tele_config = dict(
        # physical parameters
        input_image = "./stars/BHs.png", telescope_diameter_m = 6.5,
        telescope_focal_length_m = 131.4, angular_pixel_size_input_image = angular_pixel_size_input_image,
        wavelength = 100e-9, CCD_pixel_size = angular_pixel_size_input_image * 131.4 / 206265,
        CCD_pixel_count = 1024, show = False,
    )

    stars_config = dict(
        BHs_path='./224/',num_stars=0, num_BHs=1, stars_lower_size=30, stars_upper_size=50,
        height=height, width=width, bg_color=0, shape=shape, BHS_lower_size=64, BH_upper_size=75
    )

    now = datetime.datetime.now()
    date_string = now.strftime(r"%Y-%m-%d_%H-%M-%S")
    curr_dir = Path(f'logs_classification/{date_string}')
    curr_models = curr_dir / 'models'
    curr_logs = curr_dir / 'logs'
    curr_dir.mkdir()
    curr_models.mkdir()
    curr_logs.mkdir()

    writer = SummaryWriter(curr_logs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')
    csv_dir = data_dir / "labels.csv"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    dataset = []
    labels = []
    df = pd.read_csv(csv_dir)
    # df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    # df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
    try:
        df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
    except AttributeError as e:
        df['PhotoName'] = df.new_img
    df.set_index('PhotoName', inplace=True)
    T_series = df['Temperature']
    # print(df)
    images = os.listdir(data_dir)
    for i, image_name in tqdm(enumerate(images)):
        if (image_name.split('.')[1] == 'png'):
            # if i == 0:
            image_path = os.path.join(data_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            T = T_series[image_name]
            dataset.append(np.array(image))
            labels.append(T)


    Tems = np.unique(labels)
    Tems.sort()
    # Tems


    Tems_to_integer = dict((c, i) for i, c in enumerate(Tems))
    Tems_to_integer


    # labels


    labels_encoded = np.vectorize(Tems_to_integer.get)(labels)


    # labels_onehot = torch.nn.functional.one_hot(torch.tensor(labels_encoded))


    x_train, x_test, y_train, y_test = train_test_split(
        dataset, labels_encoded, test_size=0.2, random_state=2024)


    class MyData(Dataset):

        def __init__(self, dataset, labels, transform=None):
            assert len(dataset) == len(labels)
            self.transform = transform
            self.dataset = dataset
            self.labels = labels

        def __getitem__(self, idx):
            image = self.dataset[idx]
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            if self.transform:
                image = self.transform(image)
            return image, label

        def __len__(self):
            return len(self.labels)


    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4527616369984081, )*3,
                            std=(0.04355326135400156, )*3)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4527616369984081, )*3, (0.04355326135400156, )*3)
    ])
    train_dataset = MyData(x_train, y_train, transform=train_transform)
    test_dataset = MyData(x_test, y_test, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    base = torchvision.models.efficientnet_b1(weights='EfficientNet_B1_Weights.DEFAULT')
    base.classifier[0] = nn.Dropout(p=DROPOUT_RATE, inplace=True)
    base.classifier[1] = nn.Linear(in_features=1280, out_features=256, bias=True)


    class CNN(nn.Module):
        def __init__(self, base):
            super(CNN, self).__init__()
            self.base = base
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=DROPOUT_RATE)
            self.fc2 = nn.Linear(in_features=256, out_features=32, bias=True)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(p=DROPOUT_RATE)
            self.fc3 = nn.Linear(in_features=32, out_features=4, bias=True)

        def forward(self, x):
            out = self.base(x)
            out = self.relu2(out)
            out = self.dropout2(out)
            out = self.fc2(out)
            out = self.relu3(out)
            out = self.dropout3(out)
            out = self.fc3(out)

            return out


    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        torch.save(state, filename)


    def load_checkpoint(checkpoint, model, optimizer, lr):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    model = CNN(base=base)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

    test_data_size = len(test_dataset)
    train_data_size = len(train_dataset)
    test_batches_size = len(test_loader)
    total_test_step = 1
    acc_glo = 0

    name = curr_models / "final.pth.tar"
    for epoch in range(1, num_epochs + 1):
        train_losses = []
        model.train()
        loop = tqdm(train_loader, leave=False)
        for batch_idx, (X, y) in enumerate(loop):
            X = X.to(device=device)
            y = y.to(device=device)
            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = criterion(pred, y)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        loop.set_postfix(loss=loss.item())
        train_loss = sum(train_losses) / len(train_losses)
        print(f"Training Loss at epoch {epoch} is {train_loss:.4f}", end='\t')

        model.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device=device)
                y = y.to(device=device)
                pred = model(X)
                test_loss += criterion(pred, y).item()
                test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= test_batches_size
        test_acc /= test_data_size
        print(f"Testing Loss:\t{test_loss:.4f}\tAccuracy:\t{test_acc:.3f}")
        if test_acc > acc_glo and test_acc > critical_acc:
            name.unlink(missing_ok=True)
            name = curr_models / f"epoch-{epoch}-Accuracy-{test_acc:.3f}.pth.tar"
            print(f'Accuracy improve from {acc_glo:.3f} to {test_acc:.3f}, saving model dict to {name}')
            acc_glo = test_acc
            checkpoint = {"state_dict": model.state_dict(),"optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint, filename=name)
        writer.add_scalars("result/losses", {"train_loss": train_loss, "test_loss": test_loss}, total_test_step)
        writer.add_scalar("result/acc", test_acc, total_test_step)
        total_test_step += 1
    print(f'Finally best acc:{acc_glo:.3f}')
    writer.close()


    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=curr_models / "final.pth.tar")
    model.load_state_dict(torch.load(name)['state_dict'])


    test_data_size = len(test_dataset)
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        y_full = torch.tensor([]).to(device=device)
        y_pred_full = torch.tensor([]).to(device=device)
        y_pred_raw0 = torch.tensor([]).to(device=device)
        y_pred_raw1 = torch.tensor([]).to(device=device)
        y_pred_raw2 = torch.tensor([]).to(device=device)
        y_pred_raw3 = torch.tensor([]).to(device=device)
        for x, y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            y_pred_raw = model(x)
            _, y_pred = torch.max(y_pred_raw, dim=1)
            y_full = torch.cat((y_full, y), 0)
            y_pred_full = torch.cat((y_pred_full, y_pred), 0)
            y_pred_raw0 = torch.cat((y_pred_raw0, y_pred_raw[:, 0]), 0)
            y_pred_raw1 = torch.cat((y_pred_raw1, y_pred_raw[:, 1]), 0)
            y_pred_raw2 = torch.cat((y_pred_raw2, y_pred_raw[:, 2]), 0)
            y_pred_raw3 = torch.cat((y_pred_raw3, y_pred_raw[:, 3]), 0)
        loss = criterion(y_pred_full, y_full)
        total_accuracy = (y_full == y_pred_full).sum()

    print("整体测试集上的Loss: {}".format(loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))

    model.train()


    df = pd.DataFrame({'Pred': y_pred_full.squeeze().cpu().numpy(), 
                       'Real': y_full.squeeze().cpu().numpy(), 
                       'Pred_raw0': y_pred_raw0.squeeze().cpu().numpy(),
                       'Pred_raw1': y_pred_raw1.squeeze().cpu().numpy(),
                       'Pred_raw2': y_pred_raw2.squeeze().cpu().numpy(),
                       'Pred_raw3': y_pred_raw3.squeeze().cpu().numpy(),
                       })
    df.to_csv(f'{curr_dir}/acc-{acc_glo}.csv')


    cm = confusion_matrix(df.Real, df.Pred)
    df_conf = pd.DataFrame(cm)
    df_conf.to_csv(f'{curr_dir}/confusion_matrix.csv')

    plt.figure(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) + 'K' for i in Tems], )
    disp.plot(cmap='Reds')
    plt.xlabel('Predicted Temperature')
    plt.ylabel('True Temperature')
    plt.tight_layout()
    plt.savefig(f'{curr_dir}/confusion.png', dpi=600)
    plt.savefig(f'{curr_dir}/confusion.pdf')
    # plt.show()


    a = {'Model_name': 'EfficientNet-B1',
        'acc': acc_glo,
        'Batch_size': BATCH_SIZE,
        'Resolution': SIZE,
        'Dropout': DROPOUT_RATE,
        'lr': learning_rate,
        'date': date_string,
        'Training Epoch': epoch,
        'Engine': 'PyTorch',
        'angular_pixel_size_input_image': angular_pixel_size_input_image,
    }
    df = pd.read_csv('logs_classification/results.csv')
    df = pd.concat([df, pd.DataFrame([a])], ignore_index=True)
    df.to_csv('logs_classification/results.csv', index=False)
