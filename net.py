# -*- coding: utf-8 -*-
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class Network(nn.Module):
    def __init__(self, embedding_size, embedding_dimension, embedding_matrix, hidden_dimension):
        super(Network, self).__init__()
        self.hidden_dimension = hidden_dimension

        self.embedding_layer = nn.Embedding(embedding_size, embedding_dimension)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.inpt_layer_context_pre = nn.Linear(embedding_dimension, hidden_dimension)
        self.hidden_layer_context_pre = nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.inpt_layer_context_post = nn.Linear(embedding_dimension, hidden_dimension)
        self.hidden_layer_context_post = nn.Linear(hidden_dimension, hidden_dimension, bias=False)

        self.inpt_layer_np = nn.Linear(embedding_dimension, hidden_dimension)
        self.hidden_layer_np = nn.Linear(hidden_dimension, hidden_dimension)

        self.selfAttentionA = nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.selfAttentionB = nn.Linear(hidden_dimension, 1, bias=False)

        nh = hidden_dimension * 2
        self.zp_npc_layer = nn.Linear(hidden_dimension, nh)
        self.np_layer = nn.Linear(hidden_dimension, nh)
        self.nps_layer = nn.Linear(nh, nh)
        self.feature_layer = nn.Linear(61, nh)
        self.representation_hidden_layer = nn.Linear(nh, nh)
        self.output_layer = nn.Linear(nh, 1)
        self.activate = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax_layer = nn.Softmax()

        self.bi = nn.Linear(hidden_dimension, hidden_dimension)
        self.fusion = nn.Linear(hidden_dimension * 2, hidden_dimension)

    def generate_score(self, zp, npc, np, feature):
        x = self.zp_npc_layer(zp) + self.zp_npc_layer(npc) + self.np_layer(np) + self.feature_layer(feature)
        x = self.activate(x)
        x = self.representation_hidden_layer(x)
        x = self.activate(x)
        x = self.output_layer(x)
        return x

    def initHidden(self, batch=1):
        if use_cuda:
            return torch.tensor(numpy.zeros((batch, self.hidden_dimension))).type(torch.cuda.FloatTensor)
        return torch.tensor(numpy.zeros((batch, self.hidden_dimension))).type(torch.FloatTensor)

    def get_attention(self, inpt):
        return self.selfAttentionB(self.activate(self.selfAttentionA(inpt)))

    def forward_context_pre(self, word_index, hiden_layer, dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)
        word_embedding = dropout_layer(word_embedding)
        this_hidden = self.inpt_layer_context_pre(word_embedding) + self.hidden_layer_context_pre(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden

    def forward_context_post(self, word_index, hiden_layer, dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)
        word_embedding = dropout_layer(word_embedding)
        this_hidden = self.inpt_layer_context_post(word_embedding) + self.hidden_layer_context_post(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden

    def forward_np(self, word_index, hiden_layer, dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)
        word_embedding = dropout_layer(word_embedding)
        this_hidden = self.inpt_layer_np(word_embedding) + self.hidden_layer_np(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden

    def forward(self, data, dropout=0.0):
        zp_rein = torch.tensor(data["zp_rein"]).type(torch.cuda.LongTensor)
        zp_pre = torch.tensor(data["zp_pre"]).type(torch.cuda.LongTensor)
        zp_pre_mask = torch.tensor(data["zp_pre_mask"]).type(torch.cuda.FloatTensor)
        zp_post = torch.tensor(data["zp_post"]).type(torch.cuda.LongTensor)
        zp_post_mask = torch.tensor(data["zp_post_mask"]).type(torch.cuda.FloatTensor)
        candi_rein = torch.tensor(data["candi_rein"]).type(torch.cuda.LongTensor)
        np_pre = torch.tensor(data["np_pre"]).type(torch.cuda.LongTensor)
        np_pre_mask = torch.tensor(data["np_pre_mask"]).type(torch.cuda.FloatTensor)
        np_post = torch.tensor(data["np_post"]).type(torch.cuda.LongTensor)
        np_post_mask = torch.tensor(data["np_post_mask"]).type(torch.cuda.FloatTensor)
        candi = torch.tensor(data["candi"]).type(torch.cuda.LongTensor)
        candi_mask = torch.tensor(data["candi_mask"]).type(torch.cuda.FloatTensor)
        feature = torch.tensor(data["fl"]).type(torch.cuda.FloatTensor)

        zp_pre = torch.transpose(zp_pre, 0, 1)
        mask_zp_pre = torch.transpose(zp_pre_mask, 0, 1)
        hidden_zp_pre = self.initHidden()
        hiddens_zp_pre = []
        for i in range(len(mask_zp_pre)):
            hidden_zp_pre = self.forward_context_pre(zp_pre[i], hidden_zp_pre, dropout=dropout) * torch.transpose(mask_zp_pre[i:i + 1],
                                                                                                 0, 1)
            hiddens_zp_pre.append(hidden_zp_pre)
        hiddens_zp_pre = torch.cat(hiddens_zp_pre, 1)
        hiddens_zp_pre = hiddens_zp_pre.view(-1, len(mask_zp_pre), self.hidden_dimension)

        zp_post = torch.transpose(zp_post, 0, 1)
        mask_zp_post = torch.transpose(zp_post_mask, 0, 1)
        hidden_zp_post = self.initHidden()
        hiddens_zp_post = []
        for i in range(len(mask_zp_post)):
            hidden_zp_post = self.forward_context_post(zp_post[i], hidden_zp_post, dropout=dropout) * torch.transpose(
                mask_zp_post[i:i + 1],
                0, 1)
            hiddens_zp_post.append(hidden_zp_post)
        hiddens_zp_post = torch.cat(hiddens_zp_post, 1)
        hiddens_zp_post = hiddens_zp_post.view(-1, len(mask_zp_post), self.hidden_dimension)

        np_pre = torch.transpose(np_pre, 0, 1)
        mask_np_pre = torch.transpose(np_pre_mask, 0, 1)
        hidden_np_pre = self.initHidden()
        hiddens_np_pre = []
        for i in range(len(mask_np_pre)):
            hidden_np_pre = self.forward_context_pre(np_pre[i], hidden_np_pre, dropout=dropout) * torch.transpose(mask_np_pre[i:i + 1],
                                                                                                 0, 1)
            hiddens_np_pre.append(hidden_np_pre)
        hiddens_np_pre = torch.cat(hiddens_np_pre, 1)
        hiddens_np_pre = hiddens_np_pre.view(-1, len(mask_np_pre), self.hidden_dimension)

        np_post = torch.transpose(np_post, 0, 1)
        mask_np_post = torch.transpose(np_post_mask, 0, 1)
        hidden_np_post = self.initHidden()
        hiddens_np_post = []
        for i in range(len(mask_np_post)):
            hidden_np_post = self.forward_context_post(np_post[i], hidden_np_post, dropout=dropout) * torch.transpose(
                mask_np_post[i:i + 1],
                0, 1)
            hiddens_np_post.append(hidden_np_post)
        hiddens_np_post = torch.cat(hiddens_np_post, 1)
        hiddens_np_post = hiddens_np_post.view(-1, len(mask_np_post), self.hidden_dimension)

        hiddens_zp = torch.cat([hiddens_zp_pre, hiddens_zp_post], 1)
        hiddens_np = torch.cat([hiddens_np_pre, hiddens_np_post], 1)
        zp_mask = torch.cat([zp_pre_mask, zp_post_mask], 1)
        zp_mask = zp_mask[zp_rein]
        np_mask = torch.cat([np_pre_mask, np_post_mask], 1)
        np_mask = np_mask[candi_rein]
        hiddens_zp = hiddens_zp[zp_rein]
        hiddens_np = hiddens_np[candi_rein]

        Att = torch.bmm(self.relu(self.bi(hiddens_zp)), self.relu(self.bi(hiddens_np)).transpose(1, 2))

        # Mean
        Att = Att - (1 - zp_mask.unsqueeze(2)) * 1e12
        Att = Att - (1 - np_mask.unsqueeze(1)) * 1e12
        Att1 = F.softmax(Att, 2)
        Att2 = F.softmax(Att, 1).transpose(1, 2)
        bi_zps = torch.bmm(Att1, hiddens_np)
        bi_nps = torch.bmm(Att2, hiddens_zp)

        # Fusion
        zps = hiddens_zp + bi_zps
        nps = hiddens_np + bi_nps

        # self Attention
        self_Att = self.get_attention(zps)
        self_Att = self_Att - (1 - zp_mask.unsqueeze(-1)) * 1e12
        self_Att1 = F.softmax(self_Att, 1)
        average_results = torch.matmul(torch.transpose(zps, 1, 2), self_Att1)
        zp_rep = torch.sum(average_results, 2)

        self_Att = self.get_attention(nps)
        self_Att = self_Att - (1 - np_mask.unsqueeze(-1)) * 1e12
        self_Att2 = F.softmax(self_Att, 1)
        average_results = torch.matmul(torch.transpose(nps, 1, 2), self_Att2)
        npc_rep = torch.sum(average_results, 2)

        candi = torch.transpose(candi, 0, 1)
        mask_candi = torch.transpose(candi_mask, 0, 1)
        hidden_candi = self.initHidden()
        hiddens_candi = []
        for i in range(len(mask_candi)):
            hidden_candi = self.forward_np(candi[i], hidden_candi, dropout=dropout) * torch.transpose(
                mask_candi[i:i + 1], 0, 1)
            hiddens_candi.append(hidden_candi)
        hiddens_candi = torch.cat(hiddens_candi, 1)
        hiddens_candi = hiddens_candi.view(-1, len(mask_candi), self.hidden_dimension)
        hiddens_candi = hiddens_candi[candi_rein]

        Att = torch.bmm(self.relu(self.bi(hiddens_np)), self.relu(self.bi(hiddens_candi)).transpose(1, 2))
        Att = Att * np_mask.unsqueeze(2)
        Att = Att * candi_mask.unsqueeze(1)
        Att_np = torch.sum(Att, 1) / (torch.sum(np_mask, -1).unsqueeze(-1) + 1e-12)
        Att_np = Att_np - (1 - candi_mask) * 1e12
        Att_np = F.softmax(Att_np, 1).unsqueeze(-1)
        average_results = hiddens_candi * Att_np
        np_rep = torch.sum(average_results, 1)

        return zp_rep, npc_rep, np_rep, feature
