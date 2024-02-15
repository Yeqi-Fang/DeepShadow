    model2 = CNN()
    model2.to(device)
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
    df.to_csv(f'{curr_logs}/results.csv')
    name = f"{curr_models}/final.pth.tar"

    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        model2.train()
        loop = tqdm(train_loader2, leave=False)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)
            with torch.cuda.amp.autocast():
                scores = model2(data)
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
        model2.eval()
        test_loss, test_mae = 0, 0
        with torch.no_grad():
            for x, y in test_loader2:
                x = x.to(device=device)
                y = y.to(device=device)
                y_pred = model2(x)
                test_loss += criterion(y_pred, y).item()
                test_mae += torch.abs(y-y_pred).type(torch.float).sum().item()
        test_mae /= test_data_size
        test_loss /= test_batch_size
        print(f"Testing Loss:{test_loss:.4f}\tMAE:{test_mae:.3f}")
        if test_mae < mae_glo and test_mae < critical_mae:
            name = f"{curr_models}/epoch-{epoch}_MAE-{test_mae:.3f}.pth.tar"
            print(f'MAE improve from {mae_glo:.3f} to {test_mae:.3f}, saving model dict to {name}')
            mae_glo = test_mae
            checkpoint = {"state_dict": model2.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=name)
        writer.add_scalars("result/losses", {"train_loss": train_loss, "test_loss": test_loss}, step)
        writer.add_scalar("result/MAE", test_mae, step)
        df_temp = pd.DataFrame({'epoch':[epoch], 'train loss':[train_loss],
                                'test loss':[test_loss], 'test mae':[test_mae]})
        df_temp.to_csv(f'{curr_logs}/results.csv', mode='a', header=False)
        step += 1
    print(f'Finally best mae:{mae_glo:.3f}')
    writer.close()


    checkpoint = {"state_dict": model2.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint, filename=f"{curr_models}/final.pth.tar")


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
        loss = criterion(y_pred_full, y_full)
        test_mae = torch.abs(y_full-y_pred_full).type(torch.float).sum().item()

    mae = test_mae/test_data_size
    print("整体测试集上的Loss: {}".format(loss))
    print("整体测试集上的MAE: {}".format(test_mae/test_data_size))

    model2.train();
