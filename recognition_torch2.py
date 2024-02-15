import os
import cv2
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
from stars import BH_stars_img
from tqdm import tqdm
from math import ceil
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from telescope_simulator import TelescopeSimulator
from sklearn.model_selection import train_test_split
from healpy.newvisufunc import projview, newprojplot
from utils import angle_loss, LinearNDInterpolatorExt

# 14e-4, 15e-4,
# 15.5e-4, 16e-4, 1
    # 16.5e-4, 17e-4, 17.5e-4, 18e-4, 18.5e-4, 19e-4, 9.5e-4, 10.5e-4, 11.5e-4, 12.5e-4, 13.5e-4,
                                #    1.5e-4, 2.5e-4, 3.5e-4, 4.5e-4, 5.5e-4,
    # 6.5e-4, 7.5e-4, 8.5e-4, 1e-4,
    # 0.5e-4, 0.6e-4, 0.7e-4, 0.8e-4, 0.9e-4, 8.5e-4,
    #                                9.5e-4, 10.5e-4, 11.5e-4, 12.5e-4, 13.5e-4, 14.5e-4,
    #                                16.5e-4, 17e-4, 17.5e-4, 18e-4, 18.5e-4, 19e-4, 

# [6e-4, 7e-4, 8e-4, 9e-4, 10e-4, 11e-4, 12e-4, 13e-4] error
num_round = 3
angular_pixel_size_input_images = [1.1e-4 ,1.2e-4, 1.3e-4, 1.4e-4, 1.6e-4, 1.7e-4, 1.8e-4, 1.9e-4]
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
BATCH_SIZE = 300
DROPOUT_RATE = 0.5
learning_rate = 1e-3
weight_decay = 1e-4
critical_mae = 100


for angular_pixel_size_input_image in angular_pixel_size_input_images:
# try:
    if angular_pixel_size_input_image < 2e-4:
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()
    print(f'starting ----------------------{angular_pixel_size_input_image:.3e}')

    # angular_pixel_size_input_image = 4e-4

    # --------------------------------------------  Configuration ------------------------------------------------
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

    data_set = Path('tele_datasets')
    data_dirs = list(data_set.glob(f"reg_num{num_round}_*wl{wl:.3e}_*{F}*{angular_pixel_size_input_image:.2e}*_BHSize{BH_lower}-{BH_upper}"))
    assert len(data_dirs) != 0, 'Empty'
    assert len(data_dirs) == 1, "Please specify more parameters!"
    data_dir = data_dirs[0]


    now = datetime.datetime.now()
    date_string = now.strftime(r"%Y-%m-%d_%H-%M-%S")
    curr_dir = Path(f'logs_recognition/Inclination_PA/{date_string}')
    curr_models = curr_dir / 'models'
    curr_logs = curr_dir / 'logs'
    curr_dir.mkdir()
    curr_models.mkdir()
    curr_logs.mkdir()

    writer = SummaryWriter(curr_logs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_dir = data_dir / "labels.csv"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # --------------------------------------------  Data Preprocessing ------------------------------------------------
    dataset = []
    PAs = []
    Inclinations = []
    df = pd.read_csv(csv_dir)
    # df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    try:
        df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
    except AttributeError as e:
        # print(df.head())
        df['PhotoName'] = df.new_img
    df.set_index('PhotoName', inplace=True)
    PA_series = df['PA']
    inclination_series = df['Inclination']


    images = os.listdir(data_dir)
    for i, image_name in tqdm(enumerate(images)):
        if (image_name.split('.')[1] == 'png'):
            # if i == 0:
            image_path = os.path.join(data_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            PA = PA_series[image_name]
            Inclination = inclination_series[image_name]
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
    train_loader1 = DataLoader(train_dataset1, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader1 = DataLoader(test_dataset1, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    train_dataset2 = MyData(x_train2, y_train2, transform=train_transform)
    test_dataset2 = MyData(x_test2, y_test2, transform=test_transform)
    train_loader2 = DataLoader(train_dataset2, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader2 = DataLoader(test_dataset2, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)



    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            base = torchvision.models.efficientnet_b1(weights='EfficientNet_B1_Weights.DEFAULT')
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

    # --------------------------------------------  Inclination ------------------------------------------------
    model1 = CNN()
    model1.to(device)
    # loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)


    test_data_size = len(test_dataset1)
    train_data_size = len(train_dataset1)
    train_batch_size = len(train_loader1)
    test_batch_size = len(test_loader1)
    step = 1
    mae_glo_inc = 100

    df = pd.DataFrame({'epoch':[], 'train loss':[], 'test loss':[], 'test mae':[]})
    df.to_csv(curr_logs / 'results_inc.csv', index=False)
    name = curr_models / "final_inc.pth.tar"
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
        if test_mae < mae_glo_inc and test_mae < critical_mae:
            name.unlink(missing_ok=True)
            name = curr_models / f"epoch-{epoch}_MAE-{test_mae:.3f}_inc.pth.tar"
            print(f'MAE improve from {mae_glo_inc:.3f} to {test_mae:.3f}, saving model dict to {name}')
            mae_glo_inc = test_mae
            checkpoint = {"state_dict": model1.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=name)
        writer.add_scalars("result/losses", {"train_loss": train_loss, "test_loss": test_loss}, step)
        writer.add_scalar("result/MAE", test_mae, step)
        df_temp = pd.DataFrame({'epoch':[epoch], 'train loss':[train_loss],
                                'test loss':[test_loss], 'test mae':[test_mae]})
        df_temp.to_csv(curr_logs / 'results_inc.csv', mode='a', header=False)
        step += 1
    print(f'Finally best mae:{mae_glo_inc:.3f}')
    writer.close()

    checkpoint = {"state_dict": model1.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint, filename=curr_models / "final_inc.pth.tar")



    model1.load_state_dict(torch.load(name)['state_dict'])
    test_data_size = len(test_dataset1)
    model1.eval()

    with torch.no_grad():
        y_full_inc = torch.tensor([]).to(device=device)
        y_pred_full_inc = torch.tensor([]).to(device=device)
        for x, y in test_loader1:
            x = x.to(device=device)
            y = y.to(device=device)
            y_pred = model1(x)
            y_full_inc = torch.cat((y_full_inc, y), 0)
            y_pred_full_inc = torch.cat((y_pred_full_inc, y_pred), 0)
        loss = loss_fn(y_pred_full_inc, y_full_inc)
        test_mae = torch.abs(y_full_inc - y_pred_full_inc).type(torch.float).sum().item()

    mae = test_mae/test_data_size
    print("Inclination 整体测试集上的Loss: {}".format(loss))
    print("Inclination 整体测试集上的MAE: {}".format(test_mae/test_data_size))

    model1.train();
    # --------------------------------------------  Ploting ------------------------------------------------
    x = y_pred_full_inc.squeeze().cpu().numpy()
    y = y_full_inc.squeeze().cpu().numpy()
    total_range = 180
    col =[]
    sizes = []
    for i in range(0, len(x)):
        distance_to_line = abs(x[i] - y[i])
        if distance_to_line < total_range / 10: 
            col.append('blue')
            sizes.append(70)
        elif distance_to_line < total_range / 5:
            col.append('green')
            sizes.append(40)
        else: 
            col.append('magenta')
            sizes.append(40)


    plot_range = [-92, 92]
    plt.figure(figsize=(7, 7))
    # Create a line plot of the data points and the linear regression line
    plt.scatter(x, y, alpha=0.5, s=sizes, color=col)
    plt.plot(plot_range, plot_range, 'red', lw=2.5)

    # Label the axes and title the plot
    plt.xlabel(f"Predicted Inclination")
    plt.ylabel(f"Real Inclination")
    plt.xlim(*plot_range)
    plt.ylim(*plot_range)
    # plt.title("Linear Regression")
    plt.savefig(curr_dir / 'fit.png', dpi=600)
    plt.savefig(curr_dir / 'fit.pdf')

    # --------------------------------------------  PA ------------------------------------------------
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
    mae_glo_PA = 100

    df = pd.DataFrame({'epoch':[], 'train loss':[], 'test loss':[], 'test mae':[]})
    df.to_csv(curr_logs / 'results_PA.csv', index=False)
    name = curr_models / "final_PA.pth.tar"

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
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader2:
                x = x.to(device=device)
                y = y.to(device=device)
                y_pred = model2(x)
                test_loss += loss_fn(y_pred, y).item()
        test_loss /= test_batch_size
        print(f"Testing Loss and MAE:{test_loss:.4f}")
        if test_loss < mae_glo_PA and test_loss < critical_mae:
            name.unlink(missing_ok=True)
            name = curr_models / f"epoch-{epoch}_MAE-{test_loss:.3f}_PA.pth.tar"
            print(f'MAE improve from {mae_glo_PA:.3f} to {test_loss:.3f}, saving model dict to {name}')
            mae_glo_PA = test_loss
            checkpoint = {"state_dict": model2.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=name)
        writer.add_scalars("result/losses", {"train_loss": train_loss, "test_loss": test_loss}, step)
        df_temp = pd.DataFrame({'epoch':[epoch], 'train loss':[train_loss], 'test loss':[test_loss]})
        df_temp.to_csv(curr_logs / 'results_PA.csv', mode='a', header=False)
        step += 1
    print(f'Finally best mae:{mae_glo_PA:.3f}')
    writer.close()


    checkpoint = {"state_dict": model2.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint, filename=curr_models / "final_PA.pth.tar")
    model2.load_state_dict(torch.load(name)['state_dict'])


    test_data_size = len(test_dataset2)
    model2.eval()

    with torch.no_grad():
        y_full_PA = torch.tensor([]).to(device=device)
        y_pred_full_PA = torch.tensor([]).to(device=device)
        for x, y in test_loader2:
            x = x.to(device=device)
            y = y.to(device=device)
            y_pred = model2(x)
            y_full_PA = torch.cat((y_full_PA, y), 0)
            y_pred_full_PA = torch.cat((y_pred_full_PA, y_pred), 0)
        loss = loss_fn(y_pred_full_PA, y_full_PA)
    print("整体测试集上的Loss and MAE: {}".format(loss))
    model2.train();

    # --------------------------------------------  Ploting ------------------------------------------------
    pred_inc = y_pred_full_inc.squeeze().cpu().numpy()
    pred_PA = y_pred_full_PA.squeeze().cpu().numpy()
    real_inc = y_full_inc.squeeze().cpu().numpy()
    real_PA = y_full_PA.squeeze().cpu().numpy()
    df = pd.DataFrame({'Pred_inc': pred_inc, 'Pred_PA': pred_PA, 
                    'Real_inc': real_inc, 'Real_PA':real_PA})
    df.to_csv(curr_dir / f'acc-s{mae:.3f}.csv', index=False)
    err_inc = np.radians(np.abs(pred_inc - real_inc))
    real_PA = y_full_PA.squeeze()
    pred_PA = y_pred_full_PA.squeeze()
    err_PA = np.radians(angle_loss(pred_PA, real_PA).cpu().numpy())
    error = err_inc + err_PA
    nside = 32
    npix = hp.nside2npix(nside)
    thetas = np.radians(real_inc + 90)
    thetas[thetas > np.pi] = np.pi
    phis = np.radians(real_PA.cpu().numpy())
    fs = error
    the_phi = np.c_[thetas, phis]
    lut2 = LinearNDInterpolatorExt(the_phi, fs)
    N = int(10e5)
    Theta = np.random.uniform(0, np.pi, N)
    Phi = np.random.uniform(0, 2*np.pi, N)
    interpolate_points = np.zeros(N, dtype=np.float16)
    for ii in range(N):
        interpolate_points[ii] = lut2(Theta[ii], Phi[ii])
    Fs = interpolate_points

    indices = hp.ang2pix(nside, Theta, Phi)
    hpxmap = np.zeros(npix, dtype=np.float32)
    for i in range(N):
        hpxmap[indices[i]] = Fs[i]
    hp.mollview(hpxmap, title='Loss for inclination and PA')
    plt.savefig(curr_dir / 'skymap.png', dpi=600)
    plt.savefig(curr_dir / 'skymap.pdf')

    # classic healpy mollweide projections plot with graticule
    projview(
        hpxmap, coord=["G"], graticule=True, graticule_labels=True, projection_type="mollweide"
    )
    plt.savefig(curr_dir / 'skymap_grid.png', dpi=600)
    plt.savefig(curr_dir / 'skymap_grid.pdf')

    # polar view
    projview(
        hpxmap,
        coord=["G"],
        hold=False,
        graticule=True,
        graticule_labels=True,
        flip="astro",
        projection_type="polar",
        unit="cbar label",
        cb_orientation="horizontal",
        override_plot_properties={
            "cbar_shrink": 0.5,
            "cbar_pad": 0.02,
            "cbar_label_pad": -35,
            "figure_width": 16,
            "figure_size_ratio": 0.63,
        },
    );
    plt.savefig(curr_dir / 'skymap_polar.png', dpi=600)
    plt.savefig(curr_dir / 'skymap_polar.pdf')
    # --------------------------------------------  Save Results ------------------------------------------------
    a = {
        'Model_name': 'EfficientNet-B1',
        'MAE inc': mae_glo_inc,
        'MAE PA': mae_glo_PA,
        'Batch_size': BATCH_SIZE,
        'Resolution': SIZE,
        'Dropout': DROPOUT_RATE,
        'lr': learning_rate,
        'date': date_string,
        'Training Epoch': epoch,
        'Engine': 'PyTorch',
        'Loss Function': str(loss_fn),
        'angular_pixel_size_input_image': angular_pixel_size_input_image,
        'para': 'Inc, PA'
    }
    df = pd.read_csv(f'logs_recognition/results.csv')
    df = pd.concat([df, pd.DataFrame([a])], ignore_index=True)
    df.to_csv('logs_recognition/results.csv', index=False)

    # except:
    #     print(f'Failed ----------------------{angular_pixel_size_input_image:.3e}')
    #     continue