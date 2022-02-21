#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from datasets import VectorTargetDataset

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 3),
            nn.ReLU(),
        )

    def forward(self,x):
        out = self.flat(x)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size * 2,32)
        self.fc2 = nn.Linear(32,1)

    def forward(self,x):
        out = self.fc1(x)
        out = F.relu(out)
        out = F.sigmoid(self.fc2(out))

        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(3)

    #feature_encoder.apply(weights_init)
    #relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training

        # Init DataLoader from MNIST Dataset
        train_ds = VectorTargetDataset(
            MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
            dataset_seed=0,
            vector_width=2,
            gaussian_instead_of_uniform=True,
            scale=0.5,
        )  # MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
        half = len(train_ds) // 2
        train_1, train_2 = torch.utils.data.random_split(train_ds, [half, half])
        train_loader = DataLoader(train_1, batch_size=BATCH_SIZE, shuffle=True)
        train_loader2 = DataLoader(train_2, batch_size=BATCH_SIZE, shuffle=True)

        # sample datas
        samples, sample_labels = train_loader.__iter__().next()
        batches, batch_labels = train_loader2.__iter__().next()

        # calculate features
        sample_features = feature_encoder(samples.cuda(GPU)) # half x 32 ?
        batch_features = feature_encoder(batches.cuda(GPU)) # half x 32 ?

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        # (charlie) these 3 lines are just ZSL/FSL stuff, right?
        #sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        #batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        #batch_features_ext = torch.transpose(batch_features_ext,0,1)

        relation_pairs = torch.cat((sample_features,batch_features),-1)
        relations = relation_network(relation_pairs)

        # (charlie) todo get the vector labels here
        # for MNIST, cosine similarity is wonky, because all my targets are about in the same direction (noised_class, noised_class) ~ (1,1)
        # so here we will use a standard L2 norm
        target_relations = torch.norm()# F.cosine_similarity(sample_labels, batch_labels)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()


        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data[0])

        if episode%5000 == 0:

            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)

                num_per_class = 3
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                    test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(),str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy





if __name__ == '__main__':
    main()