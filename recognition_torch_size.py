import os
import cv2
import glob
import torch
import datetime
import torchvision
import numpy as np
import pandas as pd
import healpy as hp
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from math import ceil
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from stars import BH_stars_img
from torch.nn import functional as F
from scipy.interpolate import interp2d
from astrotools import auger, coord, skymap
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from telescope_simulator import TelescopeSimulator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from dataset import MyData
from utils_reg import CNN, save_checkpoint, load_checkpoint, violin_plot



sns.set_theme(style="whitegrid")
# 1.1e-4 ,1.2e-4, 1.3e-4, 
angular_pixel_size_input_images = [1.4e-4, 1.6e-4, 1.7e-4, 1.8e-4, 1.9e-4]
paras  = ['size']
num_imgaes = 3
height = 1024
width = 1024
shape = 'rect'
BH_lower = 64
BH_upper = 75
wl = 100e-9
D = 6.5
F = 131.4
SIZE = 240
num_epochs = 100
BATCH_SIZE = 256
inc_c = 91
DROPOUT_RATE = 0.2
learning_rate = 1e-3
weight_decay = 1e-4
critical_mae = 4
loss_fn = 'mse'
out_features = 1

for para in paras:
    for angular_pixel_size_input_image in angular_pixel_size_input_images:
        print(f'starting ----------------------{angular_pixel_size_input_image:.3e}')

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
        para_string = para
            
        curr_dir = Path(f'logs_recognition/{para_string}/{date_string}')
        curr_models = Path(f'logs_recognition/{para_string}/{date_string}/models')
        curr_logs = Path(f'logs_recognition/{para_string}/{date_string}/logs')
        curr_dir.mkdir()
        curr_models.mkdir()
        curr_logs.mkdir()



        writer = SummaryWriter(f"{curr_dir}/logs")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        csv_dir = f"{data_dir}/labels.csv"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


        dataset = []
        labels = []
        indexes = []
        df = pd.read_csv(csv_dir)


        # df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
        try:
            df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
        except AttributeError as e:
            # print(df.head())
            df['PhotoName'] = df.new_img
        df.set_index('PhotoName', inplace=True)
        size_series = df['size']


        images = os.listdir(data_dir)
        for i, image_name in tqdm(enumerate(images)):
            if (image_name.split('.')[1] == 'png'):
                image_path = os.path.join(data_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                # print(df_sub)
                label = size_series.loc[image_name]
                dataset.append(np.array(image))
                labels.append(label)
                indexes.append(image_name)

        # print(labels)

        x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=2024)
        _, _, index_train, index_test = train_test_split(dataset, indexes, test_size=0.2, random_state=2024)
        # pd.Series(index_train).to_csv(f'{curr_dir}/index_train.csv')
        # pd.Series(index_test).to_csv(f'{curr_dir}/index_test.csv')

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


        base = torchvision.models.efficientnet_b1(
            weights='EfficientNet_B1_Weights.IMAGENET1K_V1')
        # print(base)
        # 5, 2048, 0/1, 1280
        base.classifier[0] = nn.Dropout(p=DROPOUT_RATE, inplace=True)
        base.classifier[1] = nn.Linear(in_features=1280, out_features=256, bias=True)

        model = CNN(base=base, out_features=1, DROPOUT_RATE=DROPOUT_RATE)
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = StepLR(optimizer, step_size=10, gamma=0.2)


        test_data_size = len(test_dataset)
        train_data_size = len(train_dataset)
        train_batch_size = len(train_loader)
        test_batch_size = len(test_loader)
        step = 1
        mae_glo = 100

        df = pd.DataFrame({'epoch':[], 'train loss':[], 'test loss':[], 'test mae':[]})
        df.to_csv(f'{curr_logs}/results.csv')
        name = curr_models / "final.pth.tar"

        for epoch in range(1, num_epochs + 1):
            train_loss = 0
            model.train()
            loop = tqdm(train_loader, leave=False)
            for batch_idx, (data, targets) in enumerate(loop):
                data = data.to(device=device)
                targets = targets.to(device=device)
                with torch.cuda.amp.autocast():
                    scores = model(data)
                    loss = criterion(scores, targets)
                train_loss += loss.item()
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            loop.set_postfix(loss=loss.item())
            train_loss /= train_batch_size
            print(f"Training Loss at epoch {epoch} is {train_loss:.4f}", end='\t')
            model.eval()
            test_loss, test_mae = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device=device)
                    y = y.to(device=device)
                    y_pred = model(x)
                    test_loss += criterion(y_pred, y).item()
                    test_mae += torch.abs(y-y_pred).type(torch.float).sum().item()

            test_mae /= test_data_size
            test_loss /= test_batch_size

            print(f"Testing Loss:{test_loss:.4f}\tMAE of {para}:{test_mae:.3f}")
            
            if test_mae < mae_glo and test_mae < critical_mae:
                name.unlink(missing_ok=True)
                name = curr_models / f"epoch-{epoch}_MAE-{test_mae:.3f}.pth.tar"
                print(f'MAE improve from {mae_glo:.3f} to {test_mae:.3f}, saving model dict to {name}')
                mae_glo = test_mae
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                save_checkpoint(checkpoint, filename=name)
            writer.add_scalars("result/losses", {"train_loss": train_loss, "test_loss": test_loss}, step)
            writer.add_scalar("result/metirc", test_mae, step)
            
            df_temp = pd.DataFrame({'epoch':[epoch], 'train loss':[train_loss],
                                    'test loss':[test_loss], 'test_mae':[test_mae]})
            df_temp.to_csv(f'{curr_logs}/results.csv', mode='a', header=False)
            step += 1
        print(f'Finally best mae:{mae_glo:.3f}')
        writer.close()


        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=curr_models / "final.pth.tar")
        model.load_state_dict(torch.load(name)['state_dict'])


        model.eval()

        with torch.no_grad():
            y_full = torch.tensor([]).to(device=device)
            y_pred_full = torch.tensor([]).to(device=device)
            for x, y in test_loader:
                x = x.to(device=device)
                y = y.to(device=device)
                y_pred = model(x)
                y_full = torch.cat((y_full, y), 0)
                y_pred_full = torch.cat((y_pred_full, y_pred), 0)
            loss = criterion(y_pred_full, y_full)
            test_mae = torch.abs(y-y_pred).type(torch.float).sum().item()
            # metric is loss function it self
            mae = test_mae / test_data_size
            print(f"整体测试集上的Loss: {loss:.4f}")
            print(f"整体测试集上的MAE: {mae:.4f}")


        model.train();
        df = pd.DataFrame({'Pred': y_pred_full.squeeze().cpu().numpy(), 'Real': y_full.squeeze().cpu().numpy()})
        violin_plot(df, curr_dir, loss_fn, test_mae)
        df.to_csv(f'{curr_dir}/{loss_fn}-{test_mae:.3f}.csv')


        a = {
            'Model_name': 'EfficientNet-B1',
            'MAE': mae_glo,
            'Batch_size': BATCH_SIZE,
            'Resolution': SIZE,
            'Dropout': DROPOUT_RATE,
            'lr': learning_rate,
            'date': date_string,
            'No. training': len(index_train),
            'No. testing': len(index_test),
            'Training Epoch': epoch,
            'Engine': 'PyTorch',
            'Loss Function': loss_fn,
            'angular_pixel_size_input_image': angular_pixel_size_input_image,
            'para': para

        }
        df = pd.read_excel(f'logs_recognition/results.xlsx')
        df = pd.concat([df, pd.DataFrame([a])], ignore_index=True)
        df.to_excel('logs_recognition/results.xlsx', index=False)

