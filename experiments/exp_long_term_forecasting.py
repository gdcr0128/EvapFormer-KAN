import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import warnings
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 获取三个子序列的数据
        trend_data, trend_loader = data_provider(self.args, flag)
        seasonal_data, seasonal_loader = data_provider(self.args, flag)
        residual_data, residual_loader = data_provider(self.args, flag)
        return (trend_data, trend_loader), (seasonal_data, seasonal_loader), (residual_data, residual_loader)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, trend_loader, seasonal_loader, residual_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_trend, batch_seasonal, batch_residual) in enumerate(zip(trend_loader, seasonal_loader, residual_loader)):
                batch_trend_x, batch_trend_y = batch_trend
                batch_seasonal_x, batch_seasonal_y = batch_seasonal
                batch_residual_x, batch_residual_y = batch_residual

                batch_trend_x = batch_trend_x.float().to(self.device)
                batch_seasonal_x = batch_seasonal_x.float().to(self.device)
                batch_residual_x = batch_residual_x.float().to(self.device)

                batch_trend_y = batch_trend_y.float().to(self.device)
                batch_seasonal_y = batch_seasonal_y.float().to(self.device)
                batch_residual_y = batch_residual_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_trend_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_trend_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs_trend = self.model(batch_trend_x, dec_inp)
                outputs_seasonal = self.model(batch_seasonal_x, dec_inp)
                outputs_residual = self.model(batch_residual_x, dec_inp)

                loss_trend = criterion(outputs_trend, batch_trend_y)
                loss_seasonal = criterion(outputs_seasonal, batch_seasonal_y)
                loss_residual = criterion(outputs_residual, batch_residual_y)

                total_loss.append(loss_trend.item() + loss_seasonal.item() + loss_residual.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # 加载三个子序列的数据
        (train_trend_data, train_trend_loader), (train_seasonal_data, train_seasonal_loader), (train_residual_data, train_residual_loader) = self._get_data(flag='train')
        (vali_trend_data, vali_trend_loader), (vali_seasonal_data, vali_seasonal_loader), (vali_residual_data, vali_residual_loader) = self._get_data(flag='val')
        (test_trend_data, test_trend_loader), (test_seasonal_data, test_residual_loader), (test_residual_data, test_residual_loader) = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            # 并行遍历三个子序列的数据加载器
            for i, (batch_trend, batch_seasonal, batch_residual) in enumerate(zip(train_trend_loader, train_seasonal_loader, train_residual_loader)):
                model_optim.zero_grad()

                # 加载 trend, seasonal, residual 的数据
                batch_trend_x, batch_trend_y = batch_trend
                batch_seasonal_x, batch_seasonal_y = batch_seasonal
                batch_residual_x, batch_residual_y = batch_residual

                batch_trend_x = batch_trend_x.float().to(self.device)
                batch_seasonal_x = batch_seasonal_x.float().to(self.device)
                batch_residual_x = batch_residual_x.float().to(self.device)

                batch_trend_y = batch_trend_y.float().to(self.device)
                batch_seasonal_y = batch_seasonal_y.float().to(self.device)
                batch_residual_y = batch_residual_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_trend_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_trend_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_trend = self.model(batch_trend_x, dec_inp)
                        outputs_seasonal = self.model(batch_seasonal_x, dec_inp)
                        outputs_residual = self.model(batch_residual_x, dec_inp)
                else:
                    outputs_trend = self.model(batch_trend_x, dec_inp)
                    outputs_seasonal = self.model(batch_seasonal_x, dec_inp)
                    outputs_residual = self.model(batch_residual_x, dec_inp)

                # 计算每个子序列的损失
                loss_trend = criterion(outputs_trend, batch_trend_y)
                loss_seasonal = criterion(outputs_seasonal, batch_seasonal_y)
                loss_residual = criterion(outputs_residual, batch_residual_y)

                # 加权损失
                total_loss = (
                    self.args.lambda_trend_loss * loss_trend
                    + self.args.lambda_seasonal_loss * loss_seasonal
                    + self.args.lambda_residual_loss * loss_residual
                )

                train_loss.append(total_loss.item())

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch + 1}, Train Loss: {np.average(train_loss):.7f}")

            # 验证
            vali_loss = self.vali(vali_trend_loader, vali_seasonal_loader, vali_residual_loader, criterion)
            print(f"Epoch: {epoch + 1}, Validation Loss: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def test(self, setting, test=0):
        # 获取测试数据
        (test_trend_data, test_trend_loader), (test_seasonal_data, test_seasonal_loader), (test_residual_data, test_residual_loader) = self._get_data(flag='test')
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_trend, batch_seasonal, batch_residual) in enumerate(zip(test_trend_loader, test_seasonal_loader, test_residual_loader)):
                batch_trend_x, batch_trend_y = batch_trend
                batch_seasonal_x, batch_seasonal_y = batch_seasonal
                batch_residual_x, batch_residual_y = batch_residual

                batch_trend_x = batch_trend_x.float().to(self.device)
                batch_seasonal_x = batch_seasonal_x.float().to(self.device)
                batch_residual_x = batch_residual_x.float().to(self.device)

                batch_trend_y = batch_trend_y.float().to(self.device)
                batch_seasonal_y = batch_seasonal_y.float().to(self.device)
                batch_residual_y = batch_residual_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_trend_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_trend_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs_trend = self.model(batch_trend_x, dec_inp)
                outputs_seasonal = self.model(batch_seasonal_x, dec_inp)
                outputs_residual = self.model(batch_residual_x, dec_inp)

                outputs_trend = outputs_trend.detach().cpu().numpy()
                outputs_seasonal = outputs_seasonal.detach().cpu().numpy()
                outputs_residual = outputs_residual.detach().cpu().numpy()

                preds.append(outputs_trend)
                preds.append(outputs_seasonal)
                preds.append(outputs_residual)

                trues.append(batch_trend_y.detach().cpu().numpy())
                trues.append(batch_seasonal_y.detach().cpu().numpy())
                trues.append(batch_residual_y.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
        print(f"test shape: {preds.shape}, {trues.shape}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse: {mse}, mae: {mae}')
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_trend, batch_seasonal, batch_residual) in enumerate(pred_loader):
                batch_trend_x, batch_trend_y = batch_trend
                batch_seasonal_x, batch_seasonal_y = batch_seasonal
                batch_residual_x, batch_residual_y = batch_residual

                batch_trend_x = batch_trend_x.float().to(self.device)
                batch_seasonal_x = batch_seasonal_x.float().to(self.device)
                batch_residual_x = batch_residual_x.float().to(self.device)

                batch_trend_y = batch_trend_y.float()
                batch_seasonal_y = batch_seasonal_y.float()
                batch_residual_y = batch_residual_y.float()

                dec_inp = torch.zeros_like(batch_trend_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_trend_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs_trend = self.model(batch_trend_x, dec_inp)
                outputs_seasonal = self.model(batch_seasonal_x, dec_inp)
                outputs_residual = self.model(batch_residual_x, dec_inp)

                preds.append(outputs_trend.detach().cpu().numpy())
                preds.append(outputs_seasonal.detach().cpu().numpy())
                preds.append(outputs_residual.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

        return
