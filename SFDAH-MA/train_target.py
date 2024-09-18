import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
torch.set_printoptions(threshold=np.inf) # 为了以防万一，把这两行代码放到最前面
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader, Dataset, random_split
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import scipy.io as sio
torch.autograd.set_detect_anomaly(True)




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

# prepare data
def data_load(args):
    dset_loaders = {}
    train_bs = args.batch_size

    s_data = sio.loadmat(args.s_dset_path)
    t_data = sio.loadmat(args.t_dset_path)

    s_data_tensor = torch.from_numpy(s_data['deepfea'])
    t_data_tensor = torch.from_numpy(t_data['deepfea'])


    dim_fea = t_data_tensor.size(1)
    print("dim_fea:",dim_fea) # 4096
    s_label_tensor = torch.from_numpy(s_data['label'])
    t_label_tensor = torch.from_numpy(t_data['label'])

    if args.dset == 'office-home':
        s_label_tensor = s_label_tensor.T
        t_label_tensor = t_label_tensor.T


    source_dataset = CustomDataSet(images=s_data_tensor, labels=s_label_tensor)
    target_dataset = CustomDataSet(images=t_data_tensor, labels=t_label_tensor)


    dataset_size = len(target_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    t_train_dataset, t_test_dataset = random_split(target_dataset, [train_size, test_size])

    dset_loaders["source"] = torch.utils.data.DataLoader(source_dataset,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=4)

    dset_loaders["target_tr"]= torch.utils.data.DataLoader(t_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4)
    dset_loaders["target_te"] = torch.utils.data.DataLoader(t_test_dataset,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=4)
    dset_loaders["test"] = torch.utils.data.DataLoader(target_dataset,
                                                            batch_size=args.batch_size * 3,
                                                            shuffle=True,
                                                            num_workers=4)

    return dset_loaders, dim_fea, len(t_train_dataset)



def train_target(args):
    # load data
    dset_loaders, fea_dim, train_num = data_load(args)
    print("features dim:",fea_dim) # 4096
    print("train samples numbers:",train_num) # n_samples


    # set target network
    netB = network.fea_encoder(dim1=fea_dim).cuda()
    netC = network.feat_classifier(class_num=args.class_num, dim1=fea_dim).cuda()
    netH = network.hash_encoder(dim1=fea_dim, nbits=args.nbits).cuda()
    source_netC = network.feat_classifier(class_num=args.class_num, dim1=fea_dim).cuda()
    modelpath = args.output_dir_src + '/' + str(args.dset) + '_' + str(args.nbits) + '_' + 'source_C.pt'
    source_netC.load_state_dict(torch.load(modelpath))
    netC.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/' + str(args.dset) + '_' + str(args.nbits) + '_' + 'source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/' + str(args.dset) + '_' + str(args.nbits) + '_' + 'source_H.pt'
    netH.load_state_dict(torch.load(modelpath))



    # compute class means from source classifier as source data class-means
    source_class_means_tensor = source_netC.classifier_layer[-1].weight.data
    source_class_vars_tensor = torch.var(source_class_means_tensor, dim=0)


    # define loss function
    criterion_l2 = nn.MSELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss()

    # frozen
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False


    # Define two different optimizers
    optimizerB = optim.SGD(netB.parameters(), lr=args.lrB, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizerH = optim.SGD(netH.parameters(), lr=args.lrH, momentum=0.9, weight_decay=1e-3, nesterov=True)

    schedulerB = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerB,
                                                            T_max=args.max_epoch * len(dset_loaders["target_tr"]))
    schedulerH = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerH,
                                                            T_max=args.max_epoch * len(dset_loaders["target_tr"]))

    netB.train()
    netH.train()
    max_iter = args.max_epoch * len(dset_loaders["target_tr"]) # args.max_epoch == 200  args.batch_size = 256
    print("max_iter:", max_iter)

    interval_iter = max_iter // args.interval  #args.interval == 10

    iter_num = 0
    correct = 0
    ncount = 0

    # start train
    while iter_num < max_iter:
        class_distance_loss = 0
        t_class_means = torch.zeros([args.class_num, int(fea_dim / 2)]).cuda()
        try:
            inputs, rel_label, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target_tr"])
            inputs, rel_label, tar_idx = next(iter_test)
        if inputs.size(0) == 1:
            continue

        inputs = inputs.cuda()
        rel_label = rel_label.cuda()

        # 标签扁平化
        rel_label = rel_label.view(-1)
        target_features = netB(inputs)

        target_outputs = netC(target_features)

        target_outputs = target_outputs.float()
        target_hashcode = netH(target_features)
        p_target_pred = nn.Softmax(dim=1)(target_outputs).float()
        _, target_pred = torch.max(p_target_pred, 1)

        correct += target_pred.eq(rel_label.view_as(target_pred)).sum()
        ncount += target_pred.size(0)

        iter_num += 1

        # class moment alignment
        t_batch_class_means = compute_class_means(target_features, target_pred.long(), args.class_num)
        t_class_means += t_batch_class_means
        t_variance = torch.var(t_class_means, dim=0)
        class_distance_loss += torch.norm(source_class_means_tensor - t_class_means, p=2)
        class_distance_loss += torch.norm(source_class_vars_tensor - t_variance, p = 2)

        # semantic relationship  alignment
        normt_h_t = F.normalize((target_hashcode))
        t_h_cos = cosine_similarity(normt_h_t, normt_h_t)
        fea_t = F.normalize(target_features)
        T_f = fea_t.mm(fea_t.t())
        T_l = 2 * (sigmoid(p_target_pred.mm(p_target_pred.t()))) - 1
        image_hash_loss = criterion_l2(t_h_cos, 1.1 * T_f) + 0.9 * criterion_l2(t_h_cos, T_l).cuda()  # 1  0.9

        # quantization loss
        target_b = torch.sign(target_hashcode)
        t_sign_loss = criterion_l2(target_hashcode, target_b).cuda()



        total_loss = args.lamda1 * class_distance_loss + 10 * image_hash_loss + t_sign_loss
        # total_loss = 10 * image_hash_loss + t_sign_loss  # 消融跨域类矩对齐损失
        # total_loss = args.lamda1 * class_distance_loss + t_sign_loss  # 消融图像和哈希码语义对齐损失
        # total_loss = args.lamda1 * class_distance_loss + 10 * image_hash_loss  # 消融量化损失

        optimizerB.zero_grad()
        optimizerH.zero_grad()

        total_loss.backward()

        optimizerB.step()
        optimizerH.step()

        # Update learning rate using the scheduler
        schedulerB.step()
        schedulerH.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('Epoch [%3d/%3d]: Total Loss: %.4f '% (
                iter_num, max_iter,
                total_loss.item()
            ))
            acc_train = float(correct) * 100. / ncount
            print('Accuracy: {}/{} ({:.2f}%)'.format(correct, ncount, acc_train))
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_train)
            t_loss_str = 'loss:{}'.format(total_loss.item())
            args.out_file.write(log_str + '\n')
            args.out_file.write(t_loss_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            print(t_loss_str + '\n')

    if args.issave: # True
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        torch.save(netH.state_dict(), osp.join(args.output_dir, "target_H_" + args.savename + ".pt"))


    # test
    performance_eval(netB, netC, netH, dset_loaders["source"] , dset_loaders["target_te"])
    print('*********')
    print(args.lrB, args.lrH)
    print('*********')
    return netB, netC, netH


def performance_eval(netB, netC, netH, database_loader, query_loader):

    netB.eval().cuda()
    netC.eval().cuda()
    netH.eval().cuda()
    re_BI, re_L, qu_BI, qu_L = compress(database_loader, query_loader, netB, netC, netH)

    ## Save
    _dict = {
        'retrieval_B': re_BI,
        'L_db':re_L,
        'val_B': qu_BI,
        'L_te':qu_L,
    }
    sava_path = 'hashcode/HASH_' + args.dset + '_' + args.name + '_' + str(args.nbits) + 'bits.mat'
    if not os.path.exists(os.path.dirname(sava_path)):
        os.makedirs(os.path.dirname(sava_path))
    sio.savemat(sava_path, _dict)
    print("!!!!!!!!!DONE!!!!!!")
    return 0

def compress(database_loader, query_loader, netB, netC, netH):

    # retrieval
    re_BI = list([])
    re_L = list([])
    iter_src = iter(database_loader)
    for _, (data_I, data_L, _) in enumerate(iter_src):
        with torch.no_grad():
            var_data_I = data_I.cuda()

            code_I = netH(netB(var_data_I.to(torch.float)))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    qu_BI = list([])
    qu_L = list([])
    iter_test = iter(query_loader)
    for _, (data_I, data_L, _) in enumerate(iter_test):
        with torch.no_grad():
            var_data_I = data_I.cuda()
            code_I = netH(netB(var_data_I.to(torch.float)))
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())



    re_BI = np.array(re_BI)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_L = np.array(qu_L)

    return re_BI, re_L, qu_BI, qu_L

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def compute_class_means(x, t_psedo_labels, num_cluster):
    n_samples = x.size(0)
    if len(t_psedo_labels.size()) > 1:
        weight = t_psedo_labels.T
    else:
        weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N  [65 n_samples]
        weight[t_psedo_labels, torch.arange(n_samples)] = 1
    weight = weight.float()
    centers = torch.mm(weight, x)
    centers = F.normalize(centers, dim=1)

    return centers



def cosine_similarity(matrix1, matrix2):
    dot_product = torch.matmul(matrix1, matrix2.transpose(0, 1))
    dot_product[dot_product < 0] = 0
    norm1 = torch.norm(matrix1, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, dim=1, keepdim=True)
    similarity = dot_product / (norm1 * norm2.transpose(0, 1))
    return similarity






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFDAH-MA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--nbits', type=int, default=64, help='hash code length')

    parser.add_argument('--max_epoch', type=int, default=200, help="max iterations") # 200
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size") # 256
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['office-31', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--lrB',type=float,default=5e-4) # 5e-4
    parser.add_argument('--lrH', type=float, default=2e-2) # 2e-2
    parser.add_argument('--lamda1', type=float, default=0.1)

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
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_feature_mat.mat'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_feature_mat.mat'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())

        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())

        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par) # par_0.3
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

    elif args.dset == 'office-31':

        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_feature_mat.mat'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_feature_mat.mat'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_feature_mat.mat'
        args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper())
        # AC
        args.name = names[args.s][0].upper() + names[args.t][0].upper()
        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)  # par_0.3
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
    train_target(args)