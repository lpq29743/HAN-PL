# coding=utf8
import sys
import random
import argparse
import cPickle
import numpy
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_generator import DataGenerator
from net import Network
from utils import *


DIR = "./data/"
parser = argparse.ArgumentParser(description="Experiments\n")
parser.add_argument("-data", default=DIR, type=str, help="saved vectorized data")
parser.add_argument("-raw_data", default="./data/zp_data/", type=str, help="raw_data")

parser.add_argument("-gpu", default=6, type=int, help="GPU number")
parser.add_argument("-random_seed", default=0, type=int, help="random seed")

parser.add_argument("-epoch_num", default=200, type=int, help="epoch num")
parser.add_argument("-batch_size", default=256, type=int, help="batch size")
parser.add_argument("-learning_rate", default=5e-5, type=float, help="learning rate")
parser.add_argument("-l2_reg", default=1e-4, type=float, help="weight of L2 regularization term")
parser.add_argument("-dropout", default=0.5, type=float, help="dropout rate")
parser.add_argument("-embedding_size", default=36103, type=int, help="size of vocabulary")
parser.add_argument("-embedding_dimension", default=100, type=int, help="dimension of embedding vectors")
parser.add_argument("-hidden_dimension", default=256, type=int, help="dimension of hidden states")

parser.add_argument("-margin", default=0.1, type=float, help="margin between correct and wrong candidates")
parser.add_argument("-correct_bound", default=0.3, type=float, help="lower bound of correct candidates")
parser.add_argument("-wrong_bound", default=0.4, type=float, help="upper bound of wrong candidates")

args = parser.parse_args()

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)


def net_copy(net, copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]


def main():
    read_f = file("./data/train_data", "rb")
    train_generator = cPickle.load(read_f)
    read_f.close()
    read_f = file("./data/emb", "rb")
    embedding_matrix, _, _ = cPickle.load(read_f)
    read_f.close()
    test_generator = DataGenerator("test", args.batch_size)

    model = Network(args.embedding_size, args.embedding_dimension, embedding_matrix, args.hidden_dimension).cuda()
    best_model = Network(args.embedding_size, args.embedding_dimension, embedding_matrix, args.hidden_dimension).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)

    best_result = 0.0
    for echo in range(args.epoch_num):
        info = "[" + echo * ">" + " " * (args.epoch_num - echo) + "]"
        sys.stderr.write(info + "\r")
        cost1, cost2, cost, total_num = 0.0, 0.0, 0.0, 0
        for data in train_generator.generate_data(shuffle=True):
            zp_rep, npc_rep, np_rep, feature = model.forward(data, dropout=args.dropout)
            output = model.generate_score(zp_rep, npc_rep, np_rep, feature)
            optimizer.zero_grad()
            dis1 = output[data['wid']] - output[data['cid']] + args.margin
            dis2 = output[data['uwid']] - args.wrong_bound
            dis3 = args.correct_bound - output[data['ucid']]
            triplet_loss = torch.sum(dis1 * (dis1 > 0).cuda().float()) + torch.sum(
                dis2 * (dis2 > 0).cuda().float()) + torch.sum(dis3 * (dis3 > 0).cuda().float())

            cos_sim_sum = torch.sum(1 - F.cosine_similarity(np_rep[data['cid1']], np_rep[data['cid2']]))
            sim_w = 0.5

            num = data["result"].shape[0]

            total_loss = triplet_loss + sim_w * cos_sim_sum
            total_loss.backward()

            cost += total_loss.item() * num
            cost1 += triplet_loss.item() * num
            cost2 += cos_sim_sum.item() * num
            total_num += num
            optimizer.step()
        train_re = evaluate_train(train_generator, model)
        dev_re, dev_cost = evaluate_dev(train_generator, model, args.margin)
        if dev_re > best_result:
            best_result = dev_re
            net_copy(best_model, model)
        test_re = evaluate_test(test_generator, model)
        print 'Epoch %s; Train Cost: %.4f, %.4f, %.4f; Train Result: %.4f; Dev Result: %.4f, %.4f; Test Result: %.4f' % (
            echo, cost / total_num, cost1 / total_num, cost2 / total_num, train_re, dev_re, dev_cost, test_re)
    print >> sys.stderr
    torch.save(best_model, "./models/model")
    re = evaluate_test(test_generator, best_model)
    print "Performance on Test: F", re


def evaluate_train(generator, model):
    pr = []
    acc_num, total_num = 0, 0
    for data in generator.generate_data():
        zp_rep, npc_rep, np_rep, feature = model.forward(data)
        output = model.generate_score(zp_rep, npc_rep, np_rep, feature)
        num = data["result"].shape[0]
        total_num += num

        output = output.data.cpu().numpy()
        for s, e in data["s2e"]:
            if s == e:
                continue
            pr.append((data["result"][s:e], output[s:e]))

    predict = []
    for result, output in pr:
        index = -1
        pro = 0.0
        for i in range(len(output)):
            if output[i] > pro:
                index = i
                pro = output[i]
        predict.append(result[index])
    return sum(predict) / float(len(predict))

def evaluate_dev(generator, model, margin):
    pr = []
    acc_num, total_num, cost = 0, 0, 0.0
    for data in generator.generate_dev_data():
        zp_rep, npc_rep, np_rep, feature = model.forward(data)
        output = model.generate_score(zp_rep, npc_rep, np_rep, feature)
        dis1 = output[data['wid']] - output[data['cid']] + margin
        dis2 = output[data['uwid']] - args.wrong_bound
        dis3 = args.correct_bound - output[data['ucid']]
        triplet_loss = torch.sum(dis1 * (dis1 > 0).cuda().float()) + torch.sum(
            dis2 * (dis2 > 0).cuda().float()) + torch.sum(dis3 * (dis3 > 0).cuda().float())
        num = data["result"].shape[0]
        cost += triplet_loss.item() * num
        total_num += num

        output = output.data.cpu().numpy()
        for s, e in data["s2e"]:
            if s == e:
                continue
            pr.append((data["result"][s:e], output[s:e]))

    predict = []
    for result, output in pr:
        index = -1
        pro = 0.0
        for i in range(len(output)):
            if output[i] > pro:
                index = i
                pro = output[i]
        predict.append(result[index])
    return sum(predict) / float(len(predict)), cost / total_num


def evaluate_test(generator, model, dataset_size=1713.0):
    pr = []
    acc_num, total_num = 0, 0
    for data in generator.generate_data():
        zp_rep, npc_rep, np_rep, feature = model.forward(data)
        output = model.generate_score(zp_rep, npc_rep, np_rep, feature)

        num = data["result"].shape[0]
        total_num += num

        output = output.data.cpu().numpy()
        for s, e in data["s2e"]:
            if s == e:
                continue
            pr.append((data["result"][s:e], output[s:e]))

    predict = []
    for result, output in pr:
        index = -1
        pro = 0.0
        for i in range(len(output)):
            if output[i] > pro:
                index = i
                pro = output[i]
        predict.append(result[index])

    p = sum(predict) / float(len(predict))
    r = sum(predict) / dataset_size
    f = 0.0 if (p == 0 or r == 0) else (2.0 / (1.0 / p + 1.0 / r))
    return f


if __name__ == "__main__":
    main()
