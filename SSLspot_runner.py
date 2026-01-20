import torch
from tqdm import tqdm

import torch.nn as nn
from layer import *
from dataset import get_data_loader
import scipy.io as sio
import os
import numpy as np
import time
from blindutils import get_auc, setup_seed, TensorToHSI, init_weights
from torch import optim
from blinddataset import BS3LNetData
from torch.utils.tensorboard import SummaryWriter


class BS3LNet(nn.Module):
    def __init__(self, nch_in=189, nch_out=189, nch_ker=64, norm='bnorm', nblk=2):
        super(BS3LNet, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk
        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        self.encoder = CNR2d(self.nch_in, self.nch_ker, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm=[], relu=0.0)
        resconv = []
        for i in range(self.nblk):
            resconv += [ResBlock(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm=self.norm, relu=0.0)]
        self.resconv = nn.Sequential(*resconv)
        self.decoder = CNR2d(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm=self.norm, relu=[])
        self.conv = Conv2d(self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection')

    def forward(self, x):
        x = self.encoder(x)
        x0 = x
        x = self.resconv(x)
        x = self.decoder(x)
        x = x + x0
        x = self.conv(x)
        return x


class Trainer(object):
    '''
    Trains a model
    '''

    def __init__(self,
                 opt,
                 model,
                 criterion,
                 optimizer,
                 dataloader,
                 device,
                 model_path: str,
                 logs_path: str,
                 save_freq: int = 50,
                 scheduler=None):
        '''
        Trains a PyTorch `nn.Module` object provided in `model`
        on training sets provided in `dataloader`
        using `criterion` and `optimizer`.
        Saves model weight snapshots every `save_freq` epochs and saves the
        weights at the end of training.
        Parameters
        ----------
        model : torch model object, with callable `forward` method.
        criterion : callable taking inputs and targets, returning loss.
        optimizer : torch.optim optimizer.
        dataloader : train dataloaders.
        model_path : string. output path for model.
        logs_path : string. output path for log.
        save_freq : integer. Number of epochs between model checkpoints. Default = 50.
        scheduler : learning rate scheduler.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.model_path = model_path
        self.logs_path = logs_path
        self.save_freq = save_freq
        self.scheduler = scheduler
        self.opt = opt
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        self.log_output = open(f"{self.logs_path}/log.txt", 'w')
        self.writer = SummaryWriter(logs_path)

        print(self.opt)
        print(self.opt, file=self.log_output)

    def train_epoch(self) -> None:
        # Run a train phase for each epoch
        self.model.train(True)
        loss_train = []
        for i, data in enumerate(self.dataloader):
            label = data['label'].to(self.device)
            input = data['input'].to(self.device)
            mask = data['mask'].to(self.device)

            # forward net(batch_size=100patch=19size_window=5)
            output = self.model(
                input)  # input.shape,output.shape torch.Size([100, 204, 19, 19]) torch.Size([100, 204, 19, 19])
            # print(" output = self.model(input),input.shape,output.shape",input.shape,output.shape)
            # backward net
            self.optimizer.zero_grad()
            loss = self.criterion(output * (1 - mask), label * (1 - mask))
            loss.backward()
            self.optimizer.step()

            # get losses
            loss_train += [loss.item()]

            print("iter: " + str(i)
                  + "\tTrain Loss:" + str(round(np.mean(loss_train), 4)))
            print("iter: " + str(i)
                  + "\tTrain Loss:" + str(round(np.mean(loss_train), 4)), file=self.log_output)

        # ============ TensorBoard logging ============#
        # Log the scalar values
        info = {
            'Loss_train': np.mean(loss_train)
        }
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, self.epoch + 1)

        # Saving model
        if ((self.epoch + 1) % self.save_freq == 0):
            torch.save(self.model.state_dict(), os.path.join(self.model_path,
                                                             'BS3LNet' + '_' + self.opt.dataset + '_' + str(
                                                                 self.epoch + 1) + '.pkl'))

    def train(self) -> nn.Module:
        for epoch in range(self.opt.epochs):
            self.epoch = epoch
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs))
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs), file=self.log_output)
            print('-' * 50)
            # run training epoch
            self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
        return self.model

class SSLspot():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        self.count = args.max_sample
        self.sslbatch_size=8
        self.ssllearning_rate=1e-4
        self.sslepochs=3
        self.sslpatch=19
        self.sslsize_window=5
        self.sslratio=0.9
        self.sslgpu_ids=0
        self.seed=1
        self.dataset='mvtec 3D'
    def train_model(opt,class_name):
        print(f'\n\ntrain_model on class {class_name}...')
        DB = 'mvtec 3D'
        expr_dir = os.path.join('./checkpoints/', DB)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        prefix = 'BS3LNet' + '_batch_size_' + str(opt.sslbatch_size) + '_epoch_' + str(
            opt.sslepochs) + '_learning_rate_' + str(opt.ssllearning_rate) + \
                 '_patch_' + str(opt.sslpatch) + '_size_window_' + str(opt.sslsize_window) + '_ratio_' + str(
            opt.sslratio) + '_gpu_ids_' + str(opt.sslgpu_ids)
        trainfile = os.path.join(expr_dir, prefix)
        if not os.path.exists(trainfile):
            os.makedirs(trainfile)
        # Device
        device = torch.device('cuda:{}'.format(opt.sslgpu_ids)) if torch.cuda.is_available() else torch.device('cpu')
        # Directories for storing model and loss
        model_path = os.path.join(trainfile, 'model')
        logs_path = os.path.join(trainfile, './logs')
        setup_seed(opt.seed)
        loader_train, band = BS3LNetData(opt)
        net = BS3LNet(band, band, nch_ker=opt.nch_ker, norm=opt.norm_mode, nblk=opt.nblk).to(device)
        # Initialize net parameters
        init_weights(net, init_type=opt.init_weight_type, init_gain=opt.init_gain)
        # Define Optimizers and Loss
        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999),
                               weight_decay=opt.weight_decay)
        scheduler_net = None

        if opt.lossm.lower() == 'l1':
            criterion = nn.L1Loss().to(device)  # Regression loss: L1
        elif opt.lossm.lower() == 'l2':
            criterion = nn.MSELoss().to(device)  # Regression loss: L2

        if torch.cuda.is_available():
            print('Model moved to CUDA compute device.')
        else:
            print('No CUDA available, running on CPU!')
        # Training
        t_begin = time.time()
        trainer = Trainer(opt,
                          net,
                          criterion,
                          optimizer,
                          loader_train,
                          device,
                          model_path,
                          logs_path,
                          scheduler=scheduler_net)
        trainer.train()
        t_end = time.time()
        print('Time of training-{}s'.format((t_end - t_begin)))

    # def evaluate(self, class_name):
    #     image_rocaucs = dict()
    #     pixel_rocaucs = dict()
    #     au_pros = dict()
    #     test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)
    #     path_list = []
    #     with torch.no_grad():
    #         for sample, mask, label, rgb_path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
    #             for method in self.methods.values():
    #                 method.predict(sample, mask, label)
    #                 path_list.append(rgb_path)
    #
    #     for method_name, method in self.methods.items():
    #         method.calculate_metrics()
    #         image_rocaucs[method_name] = round(method.image_rocauc, 3)
    #         pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
    #         au_pros[method_name] = round(method.au_pro, 3)
    #         print(
    #             f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
    #         if self.args.save_preds:
    #             method.save_prediction_maps('./pred_maps', path_list)
    #     return image_rocaucs, pixel_rocaucs, au_pros

    def evaluate(opt, class_name):
        print(f'\n\ntest_model on class {class_name}...')
        DB = 'mvtec 3D'
        expr_dir = os.path.join('./checkpoints/', DB)
        prefix = 'BS3LNet' + '_batch_size_' + str(opt.sslbatch_size) + '_epoch_' + str(
            opt.sslepochs) + '_learning_rate_' + str(opt.ssllearning_rate) + \
                 '_patch_' + str(opt.sslpatch) + '_size_window_' + str(opt.sslsize_window) + '_ratio_' + str(
            opt.sslratio) + '_gpu_ids_' + str(opt.sslgpu_ids)
        trainfile = os.path.join(expr_dir, prefix)
        model_path = os.path.join(trainfile, 'model')
        expr_dirs = os.path.join('./result/', DB)
        if not os.path.exists(expr_dirs):
            os.makedirs(expr_dirs)
        log_output = open(f"{expr_dirs}/log.txt", 'w')
        model_weights = os.path.join(model_path, 'BS3LNet' + '_' + opt.dataset + '_' + str(opt.epochs) + '.pkl')
        # test datalodar
        data_dir = './data/'
        image_file = data_dir + opt.dataset + '.mat'
        input_data = sio.loadmat(image_file)
        image = input_data['data']
        image = image.astype(np.float32)
        gt = input_data['map']
        gt = gt.astype(np.float32)
        band = image.shape[2]
        test_data = np.expand_dims(image, axis=0)
        loader_test = torch.from_numpy(test_data.transpose(0, 3, 1, 2)).type(torch.FloatTensor)

        # Device
        device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
        net = BS3LNet(band, band, nch_ker=opt.nch_ker, norm=opt.norm_mode, nblk=opt.nblk).to(device)
        net.load_state_dict(torch.load(model_weights, map_location='cuda:0'))
        t_begin = time.time()
        net.eval()
        test_data = loader_test
        img_old = test_data.to(device)
        img_new = net(img_old)
        HSI_old = TensorToHSI(img_old)
        HSI_new = TensorToHSI(img_new)
        auc, detectmap = get_auc(HSI_old, HSI_new, gt)
        t_end = time.time()
        print("AUC: " + str(auc))
        print("AUC: " + str(auc), file=log_output)
        print('Time of testing-{}s'.format((t_end - t_begin)))
        print('Time of testing-{}s'.format((t_end - t_begin)), file=log_output)
        sio.savemat(os.path.join(expr_dirs, 'detectmap.mat'), {'detectmap': detectmap})
        sio.savemat(os.path.join(expr_dirs, 'reconstructed_data.mat'), {'reconstructed_data': HSI_new})