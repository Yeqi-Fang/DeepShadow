import os
import cv2
import glob
import torch
import datetime
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from stars import BH_stars_img
from telescope_simulator import TelescopeSimulator
from tqdm import tqdm
from math import ceil
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split



def angle_loss(output, target):
    # output_angle = output * torch.pi / 180
    # target_angle = target * torch.pi / 180
    # loss = torch.mean((torch.cos(output_angle) - torch.cos(target_angle))**2 + \
    #                   (torch.sin(output_angle) - torch.sin(target_angle))**2)
    loss = torch.mean(torch.min(torch.abs(output - target), torch.abs(360 + output - target)))
    return loss


angular_pixel_size_input_images = [16.5e-4]
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
# IN_SIZE = 8
loss_fn = 'mse' # angle
num_epochs = 100
BATCH_SIZE = 256
inc_c = 30
DROPOUT_RATE = 0.5
learning_rate = 1e-3
weight_decay = 1e-4
critical_mae = 20


for angular_pixel_size_input_image in angular_pixel_size_input_images:
    print(f'starting ----------------------{angular_pixel_size_input_image:.3e}')

    # angular_pixel_size_input_image = 4e-4


    tele_config = dict(
        # physical parameters
        input_image = r"./stars/BHs.png", telescope_diameter_m = 6.5,
        telescope_focal_length_m = 131.4, angular_pixel_size_input_image = angular_pixel_size_input_image,
        wavelength = 100e-9, CCD_pixel_size = angular_pixel_size_input_image * 131.4 / 206265,
        CCD_pixel_count = 1024, show = False,
    )

    stars_config = dict(
        BHs_path='./224/',num_stars=0, num_BHs=1, stars_lower_size=30, stars_upper_size=50,
        height=height, width=width, bg_color=0, shape=shape, BHS_lower_size=64, BH_upper_size=75
    )


    data_dirs = glob.glob(f"tele_datasets/reg_num{num_imgaes}_rect_wl{wl:.3e}_*{F}*{angular_pixel_size_input_image:.2e}*_BHSize{BH_lower}-{BH_upper}")
    assert len(data_dirs) != 0, 'Empty'
    assert len(data_dirs) == 1, "Please specify more parameters!"
    data_dir = data_dirs[0]


    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(f'logs_recognition/Inclination_PA/{date_string}')
    os.mkdir(f'logs_recognition/Inclination_PA/{date_string}/models')
    os.mkdir(f'logs_recognition/Inclination_PA/{date_string}/logs')
    curr_dir = f'logs_recognition/Inclination_PA/{date_string}'
    curr_models = f'logs_recognition/Inclination_PA/{date_string}/models'
    curr_logs = f'logs_recognition/Inclination_PA/{date_string}/logs'



    writer = SummaryWriter(f"{curr_dir}/logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_dir = f"{data_dir}/labels.csv"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    dataset = []
    PAs = []
    Inclinations = []
    df = pd.read_csv(csv_dir)


    # df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
    df.set_index('PhotoName', inplace=True)
    PA = df['PA']
    inclination = df['Inclination']


    images = os.listdir(data_dir)
    for i, image_name in tqdm(enumerate(images)):
        if (image_name.split('.')[1] == 'png'):
            # if i == 0:
            image_path = os.path.join(data_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # image = cv2.resize(image, (SIZE, SIZE))
            # print(image_name)
            # stars_config['BHs'] = image_name
            # img = BH_stars_img(**stars_config)
            # img.stars_gen()
            # img.stars_gen()
            # img.BHs_gen()
            # noise_BHs = img.add_noise(img.stars_BHs_img, radius=0)
            # tele_config['input_image'] = noise_BHs
            # telescope_simulator = TelescopeSimulator(**tele_config)
            # output_img = telescope_simulator.generate_image(show=False)
            # x = np.random.randint(0, background.shape[1] - img64.shape[1])
            # y = np.random.randint(0, background.shape[0] - img64.shape[0])
            # new = background.copy()
            # new[y:y+img64.shape[0], x:x+img64.shape[1]] = img64
            PA = PA[image_name]
            Inclination = inclination[image_name]
            dataset.append(np.array(image))
            PAs.append(PA)
            Inclinations.append(Inclination)

    x_train1, x_test1, y_train1, y_test1 = train_test_split(dataset, Inclinations, test_size=0.2, random_state=2024)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(dataset, PAs, test_size=0.2, random_state=2024)


    class MyData(Dataset):

        def __init__(self, dataset, labels, transform=None):
            assert len(dataset) == len(labels)
            self.transform = transform
            self.dataset = dataset
            self.labels = labels

        def __getitem__(self, idx):
            image = self.dataset[idx]
            label = torch.tensor([self.labels[idx]], dtype=torch.float32)
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
    
    train_dataset1 = MyData(x_train1, y_train1, transform=train_transform)
    test_dataset1 = MyData(x_test1, y_test1, transform=test_transform)
    train_loader1 = DataLoader(train_dataset1, batch_size=BATCH_SIZE, shuffle=False)
    test_loader1 = DataLoader(test_dataset1, batch_size=BATCH_SIZE, shuffle=False)

    train_dataset2 = MyData(x_train2, y_train2, transform=train_transform)
    test_dataset2 = MyData(x_test2, y_test2, transform=test_transform)
    train_loader2 = DataLoader(train_dataset2, batch_size=BATCH_SIZE, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=BATCH_SIZE, shuffle=False)



    class CNN(nn.Module):
        def __init__(self, base):
            super(CNN, self).__init__()
            base = torchvision.models.efficientnet_b1(weights='EfficientNet_B1_Weights.IMAGENET1K_V1')
            base.classifier[0] = nn.Dropout(p=DROPOUT_RATE, inplace=True)
            base.classifier[1] = nn.Linear(in_features=1280, out_features=256, bias=True)
            self.base = base
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=DROPOUT_RATE)
            self.fc2 = nn.Linear(in_features=256, out_features=32, bias=True)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(p=DROPOUT_RATE)
            self.fc3 = nn.Linear(in_features=32, out_features=1, bias=True)

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


    model1 = CNN()
    model1.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)


    test_data_size = len(test_dataset1)
    train_data_size = len(train_dataset1)
    train_batch_size = len(train_loader1)
    test_batch_size = len(test_loader1)
    step = 1
    mae_glo = 100

    df = pd.DataFrame({'epoch':[], 'train loss':[], 'test loss':[], 'test mae':[]})
    df.to_csv(f'{curr_logs}/results_inc.csv')
    name = f"{curr_models}/final_inc.pth.tar"

    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        model1.train()
        loop = tqdm(train_loader1, leave=False)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)
            with torch.cuda.amp.autocast():
                scores = model1(data)
                loss = loss_fn(scores, targets)
            train_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        loop.set_postfix(loss=loss.item())
        train_loss /= train_batch_size
        print(f"Training Loss at epoch {epoch} is {train_loss:.4f}", end='\t')
        model1.eval()
        test_loss, test_mae = 0, 0
        with torch.no_grad():
            for x, y in test_loader1:
                x = x.to(device=device)
                y = y.to(device=device)
                y_pred = model1(x)
                test_loss += loss_fn(y_pred, y).item()
                test_mae += torch.abs(y-y_pred).type(torch.float).sum().item()
        test_mae /= test_data_size
        test_loss /= test_batch_size
        print(f"Testing Loss:{test_loss:.4f}\tMAE:{test_mae:.3f}")
        if test_mae < mae_glo and test_mae < critical_mae:
            name = f"{curr_models}/epoch-{epoch}_MAE-{test_mae:.3f}_inc.pth.tar"
            print(f'MAE improve from {mae_glo:.3f} to {test_mae:.3f}, saving model dict to {name}')
            mae_glo = test_mae
            checkpoint = {"state_dict": model1.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=name)
        writer.add_scalars("result/losses", {"train_loss": train_loss, "test_loss": test_loss}, step)
        writer.add_scalar("result/MAE", test_mae, step)
        df_temp = pd.DataFrame({'epoch':[epoch], 'train loss':[train_loss],
                                'test loss':[test_loss], 'test mae':[test_mae]})
        df_temp.to_csv(f'{curr_logs}/results_inc.csv', mode='a', header=False)
        step += 1
    print(f'Finally best mae:{mae_glo:.3f}')
    writer.close()


    checkpoint = {"state_dict": model1.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint, filename=f"{curr_models}/final_inc.pth.tar")


    # best_model = name
    # path_of_best_model = os.path.join(curr_models, best_model)
    model1.load_state_dict(torch.load(name)['state_dict'])


    test_data_size = len(test_dataset1)
    model1.eval()

    with torch.no_grad():
        y_full = torch.tensor([]).to(device=device)
        y_pred_full = torch.tensor([]).to(device=device)
        for x, y in test_loader1:
            x = x.to(device=device)
            y = y.to(device=device)
            y_pred = model1(x)
            y_full = torch.cat((y_full, y), 0)
            y_pred_full = torch.cat((y_pred_full, y_pred), 0)
        loss = loss_fn(y_pred_full, y_full)
        test_mae = torch.abs(y_full-y_pred_full).type(torch.float).sum().item()

    mae = test_mae/test_data_size
    print("整体测试集上的Loss: {}".format(loss))
    print("整体测试集上的MAE: {}".format(test_mae/test_data_size))

    model1.train();

    model2 = CNN()
    model2.to(device)
    loss_fn = angle_loss
    optimizer = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

    test_data_size = len(test_dataset2)
    train_data_size = len(train_dataset2)
    train_batch_size = len(train_loader2)
    test_batch_size = len(test_loader2)
    step = 1
    mae_glo = 100

    df = pd.DataFrame({'epoch':[], 'train loss':[], 'test loss':[], 'test mae':[]})
    df.to_csv(f'{curr_logs}/results_PA.csv')
    name = f"{curr_models}/final_PA.pth.tar"

    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        model2.train()
        loop = tqdm(train_loader2, leave=False)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)
            with torch.cuda.amp.autocast():
                scores = model2(data)
                loss = loss_fn(scores, targets)
            train_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        loop.set_postfix(loss=loss.item())
        train_loss /= train_batch_size
        print(f"Training Loss at epoch {epoch} is {train_loss:.4f}", end='\t')
        model2.eval()
        test_loss, test_mae = 0, 0
        with torch.no_grad():
            for x, y in test_loader2:
                x = x.to(device=device)
                y = y.to(device=device)
                y_pred = model2(x)
                test_loss += loss_fn(y_pred, y).item()
                test_mae += torch.abs(y-y_pred).type(torch.float).sum().item()
        test_mae /= test_data_size
        test_loss /= test_batch_size
        print(f"Testing Loss:{test_loss:.4f}\tMAE:{test_mae:.3f}")
        if test_mae < mae_glo and test_mae < critical_mae:
            name = f"{curr_models}/epoch-{epoch}_MAE-{test_mae:.3f}_PA.pth.tar"
            print(f'MAE improve from {mae_glo:.3f} to {test_mae:.3f}, saving model dict to {name}')
            mae_glo = test_mae
            checkpoint = {"state_dict": model2.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=name)
        writer.add_scalars("result/losses", {"train_loss": train_loss, "test_loss": test_loss}, step)
        writer.add_scalar("result/MAE", test_mae, step)
        df_temp = pd.DataFrame({'epoch':[epoch], 'train loss':[train_loss],
                                'test loss':[test_loss], 'test mae':[test_mae]})
        df_temp.to_csv(f'{curr_logs}/results_PA.csv', mode='a', header=False)
        step += 1
    print(f'Finally best mae:{mae_glo:.3f}')
    writer.close()


    checkpoint = {"state_dict": model2.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint, filename=f"{curr_models}/final_PA.pth.tar")


    # best_model = name
    # path_of_best_model = os.path.join(curr_models, best_model)
    model2.load_state_dict(torch.load(name)['state_dict'])


    test_data_size = len(test_dataset2)
    model2.eval()

    with torch.no_grad():
        y_full = torch.tensor([]).to(device=device)
        y_pred_full = torch.tensor([]).to(device=device)
        for x, y in test_loader2:
            x = x.to(device=device)
            y = y.to(device=device)
            y_pred = model2(x)
            y_full = torch.cat((y_full, y), 0)
            y_pred_full = torch.cat((y_pred_full, y_pred), 0)
        loss = loss_fn(y_pred_full, y_full)
        test_mae = torch.abs(y_full-y_pred_full).type(torch.float).sum().item()

    mae = test_mae/test_data_size
    print("整体测试集上的Loss: {}".format(loss))
    print("整体测试集上的MAE: {}".format(test_mae/test_data_size))

    model2.train();



    df = pd.DataFrame({'Pred': y_pred_full.squeeze().cpu().numpy(), 'Real': y_full.squeeze().cpu().numpy()})
    df.to_csv(f'{curr_dir}/acc:{mae:.3f}.csv')



    a = {
        'Model_name': 'EfficientNet-B1',
        'MAE': mae_glo,
        'Batch_size': BATCH_SIZE,
        # 'In size': IN_SIZE,
        'Resolution': SIZE,
        'Dropout': DROPOUT_RATE,
        'lr': learning_rate,
        'date': date_string,
        'Training Epoch': epoch,
        'Engine': 'PyTorch',
        'Loss Function': loss_fn,
        'angular_pixel_size_input_image': angular_pixel_size_input_image,
        'para': 'Inc, PA'

    }
    df = pd.read_excel(f'logs_recognition/results.xlsx')
    df = pd.concat([df, pd.DataFrame([a])], ignore_index=True)
    df.to_excel('logs_recognition/results.xlsx', index=False)

