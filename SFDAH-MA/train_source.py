import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network, loss
from torch.utils.data import DataLoader, Dataset, random_split
import random
from loss import CrossEntropyLabelSmooth
import torch.nn.functional as F
import scipy.io as sio


class CustomDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

# returns a tuple containing the image, label, and the index itself
    def __getitem__(self, index):
        return self.images[index], self.labels[index], index

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count

def data_load(args):
    ## prepare data
    dset_loaders = {}
    s_data = sio.loadmat(args.s_dset_path)
    t_data = sio.loadmat(args.test_dset_path)
    s_data_tensor = torch.from_numpy(s_data['deepfea'])
    t_data_tensor = torch.from_numpy(t_data['deepfea'])

    dim_fea = s_data_tensor.size(1)
    s_label_tensor = torch.from_numpy(s_data['label'])
    t_label_tensor = torch.from_numpy(t_data['label'])

    if args.dset == 'office-home':
        s_label_tensor = s_label_tensor.T
        t_label_tensor = t_label_tensor.T

    source_dataset = CustomDataSet(images=s_data_tensor, labels=s_label_tensor)
    target_dataset = CustomDataSet(images=t_data_tensor, labels=t_label_tensor)
    dataset_size = len(source_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    s_train_dataset, s_test_dataset = random_split(source_dataset, [train_size, test_size])

    dset_loaders["source"] = torch.utils.data.DataLoader(source_dataset,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=4)

    dset_loaders["source_tr"]= torch.utils.data.DataLoader(s_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4)
    dset_loaders["source_te"] = torch.utils.data.DataLoader(s_test_dataset,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=4)
    dset_loaders["test"] = torch.utils.data.DataLoader(target_dataset,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=4)


    return dset_loaders, dim_fea



def train_source(args):
    dset_loaders, fea_dim = data_load(args)
    netB = network.fea_encoder(dim1=fea_dim).cuda()
    netC = network.feat_classifier(class_num=args.class_num, dim1=fea_dim).cuda()
    netH = network.hash_encoder(dim1=fea_dim, nbits=args.nbits).cuda()

    criterion_l2 = nn.MSELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()


    # Define optimizers
    optimizerB = optim.SGD(netB.parameters(), lr=args.lrB, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizerC = optim.SGD(netC.parameters(), lr=args.lrC, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizerH = optim.SGD(netH.parameters(), lr=args.lrH, momentum=0.9, weight_decay=1e-3, nesterov=True)

    # Initialize CosineAnnealingLR schedulers
    schedulerB = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerB,
                                                            T_max=args.max_epoch * len(dset_loaders["source_tr"]))
    schedulerC = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerC,
                                                            T_max=args.max_epoch * len(dset_loaders["source_tr"]))

    schedulerH = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerH,
                                                            T_max=args.max_epoch * len(dset_loaders["source_tr"]))

    # 初始化精度和损失
    acc_init = 0
    loss_init = 10000
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10 # interval_iter = 70
    iter_num = 0
    netB.train()
    netC.train()
    netH.train()

    correct = 0
    ncount = 0

    while iter_num < max_iter:
        try:
            inputs_source, labels_source, index_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source, index_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        labels_source = labels_source.view(-1)

        batch_fea = netB(inputs_source)

        outputs_source = netC(batch_fea)

        source_hash = netH(batch_fea)

        # for acc
        preds = outputs_source.max(1, keepdim=True)[1]  # 标签
        # print(preds.shape)
        correct += preds.eq(labels_source.view_as(preds)).sum()
        ncount += preds.size(0)


        # 1. loss1
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)


        device = labels_source.device
        label_source_onehot = torch.eye(args.class_num, device=device)[labels_source, :]
        S_I = label_source_onehot.mm(label_source_onehot.t())
        S_I = S_I.cuda()  # [0, 1]
        h_norm_s = F.normalize(source_hash)
        S_h_s = h_norm_s.mm(h_norm_s.t())  # [-1, 1]
        S_h_s[S_h_s < 0] = 0  # [0, 1]
        relation_recons_loss = criterion_l2(S_h_s, 1.1 * S_I)

        source_b = torch.sign(source_hash)
        s_sign_loss = criterion_l2(source_hash, source_b)

        source_loss = classifier_loss + 0.9 * relation_recons_loss + 0.1 * s_sign_loss


        optimizerB.zero_grad()
        optimizerC.zero_grad()
        optimizerH.zero_grad()
        source_loss.backward()

        optimizerB.step()
        optimizerC.step()
        optimizerH.step()
        # Update learning rate using the scheduler
        schedulerB.step()
        schedulerC.step()
        schedulerH.step()


        if iter_num % interval_iter == 0 or iter_num == max_iter:

            print('Epoch [%3d/%3d]: Total Loss: %.4f, loss_cl: %.4f, loss_sim: %.4f, loss_qu: %.4f' % (
            iter_num, max_iter,
            source_loss.item(),
            classifier_loss.item(),
            relation_recons_loss.item(),
            s_sign_loss.item()
            ))

            acc_train = float(correct) * 100. / ncount
            print('Accuracy: {}/{} ({:.2f}%)'.format(correct, ncount, acc_train))
            if acc_train >= acc_init:
                acc_init = acc_train
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            if source_loss.item() <= loss_init:
                loss_init = source_loss.item()
                best_netH = netH.state_dict()

            netB.train()
            netC.train()
            netH.train()

    torch.save(best_netB, osp.join(args.output_dir_src, str(args.dset) + '_' + str(args.nbits) + '_' + "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, str(args.dset) + '_' + str(args.nbits) + '_' + "source_C.pt"))
    torch.save(best_netH, osp.join(args.output_dir_src, str(args.dset) + '_' + str(args.nbits) + '_' + "source_H.pt"))

    return dset_loaders




def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFDAH-MA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--nbits',type=int,default=32,help='hash code length')
    parser.add_argument('--lamda', type=float, default=0.9)  # 1

    parser.add_argument('--max_epoch', type=int, default=600, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-A', 'office-31', 'office-home'])

    parser.add_argument('--lrB', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--lrC', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--lrH', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed") # 2020
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])


    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office-31':
        names = ['Amazon', 'Dslr', 'Webcam']
        args.class_num = 31


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = './data/'
    if args.dset == 'office-home':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_feature_mat.mat'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_feature_mat.mat'


        args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
        # print(args.output_dir_src)
        args.name_src = names[args.s][0].upper()
        if not osp.exists(args.output_dir_src):
            os.system('mkdir -p ' + args.output_dir_src)
        if not osp.exists(args.output_dir_src):
            os.mkdir(args.output_dir_src)

        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

    if args.dset == 'office-31':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_feature_mat.mat'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_feature_mat.mat'

        # san/uda/office-home/A
        args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
        args.name_src = names[args.s][0].upper()  # A
        # create path
        if not osp.exists(args.output_dir_src):
            os.system('mkdir -p ' + args.output_dir_src)
        if not osp.exists(args.output_dir_src):
            os.mkdir(args.output_dir_src)

        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()


    if args.dset == 'office-home' and args.s == 0:
        args.max_epoch = 1000
    dset_loaders = train_source(args)

    print("{} Source model trained complete!!!".format(names[args.s]))

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
