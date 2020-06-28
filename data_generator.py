# coding=utf8
import argparse
import cPickle
import numpy
import random

DIR = "./data/"
parser = argparse.ArgumentParser(description="Experiments\n")
parser.add_argument("-data", default=DIR, type=str, help="saved vectorized data")
parser.add_argument("-raw_data", default="./data/zp_data/", type=str, help="raw_data")
parser.add_argument("-random_seed", default=0, type=int, help="random seed")

args = parser.parse_args()


class DataGenerator:
    def __init__(self, file_type, max_pair):
        data_path = args.data + file_type + "/"
        self.candi_vec = numpy.load(data_path + "candi_vec.npy")
        self.candi_vec_mask = numpy.load(data_path + "candi_vec_mask.npy")

        self.np_post = numpy.load(data_path + "np_post.npy")
        self.np_post_vec_mask = numpy.load(data_path + "np_post_mask.npy")
        self.np_pre = numpy.load(data_path + "np_pre.npy")
        self.np_pre_vec_mask = numpy.load(data_path + "np_pre_mask.npy")

        self.zp_post = numpy.load(data_path + "zp_post.npy")
        self.zp_post_vec_mask = numpy.load(data_path + "zp_post_mask.npy")
        self.zp_pre = numpy.load(data_path + "zp_pre.npy")
        self.zp_pre_vec_mask = numpy.load(data_path + "zp_pre_mask.npy")

        self.ifl_vec = numpy.load(data_path + "ifl_vec.npy")

        read_f = file(data_path + "zp_candi_pair_info", "rb")
        zp_candis_pair = cPickle.load(read_f)
        read_f.close()

        self.data_batch = []
        zp_rein = []
        candi_rein = []
        this_target = []
        this_result = []
        s2e = []
        for i in range(len(zp_candis_pair)):
            zpi, candis = zp_candis_pair[i]
            if len(candis) + len(candi_rein) > max_pair and len(candi_rein) > 0:
                ci_s = candi_rein[0]
                ci_e = candi_rein[-1] + 1
                zpi_s = zp_rein[0]
                zpi_e = zp_rein[-1] + 1
                this_batch = {}
                this_batch["zp_rein"] = numpy.array(zp_rein, dtype="int32") - zp_rein[0]
                this_batch["candi_rein"] = numpy.array(candi_rein, dtype="int32") - candi_rein[0]
                this_batch["target"] = numpy.array(this_target, dtype="int32")
                this_batch["result"] = numpy.array(this_result, dtype="int32")
                this_batch["zp_post"] = self.zp_post[zpi_s:zpi_e]
                this_batch["zp_pre"] = self.zp_pre[zpi_s:zpi_e]
                this_batch["zp_post_mask"] = self.zp_post_vec_mask[zpi_s:zpi_e]
                this_batch["zp_pre_mask"] = self.zp_pre_vec_mask[zpi_s:zpi_e]
                this_batch["candi"] = self.candi_vec[ci_s:ci_e]
                this_batch["candi_mask"] = self.candi_vec_mask[ci_s:ci_e]
                this_batch["np_post"] = self.np_post[ci_s:ci_e]
                this_batch["np_pre"] = self.np_pre[ci_s:ci_e]
                this_batch["np_post_mask"] = self.np_post_vec_mask[ci_s:ci_e]
                this_batch["np_pre_mask"] = self.np_pre_vec_mask[ci_s:ci_e]
                this_batch["fl"] = self.ifl_vec[ci_s:ci_e]
                # this_batch["infos"] = self.infos[ci_s:ci_e]
                this_batch["s2e"] = s2e
                correct_indexs, wrong_indexs, uni_correct_indexs, uni_wrong_indexs, correct_indexs1, correct_indexs2 = self.get_index(
                    this_batch['s2e'], this_batch['result'])
                this_batch['cid'] = correct_indexs
                this_batch['wid'] = wrong_indexs
                this_batch['ucid'] = uni_correct_indexs
                this_batch['uwid'] = uni_wrong_indexs
                this_batch['cid1'] = correct_indexs1
                this_batch['cid2'] = correct_indexs2
                self.data_batch.append(this_batch)
                zp_rein = []
                candi_rein = []
                this_target = []
                this_result = []
                s2e = []
            start = len(this_result)
            end = start
            for candii, res, tar in candis:
                zp_rein.append(zpi)
                candi_rein.append(candii)
                this_target.append(tar)
                this_result.append(res)
                end += 1
            s2e.append((start, end))
        if len(candi_rein) > 0:
            ci_s = candi_rein[0]
            ci_e = candi_rein[-1] + 1
            zpi_s = zp_rein[0]
            zpi_e = zp_rein[-1] + 1
            this_batch = {}
            this_batch["zp_rein"] = numpy.array(zp_rein, dtype="int32") - zp_rein[0]
            this_batch["candi_rein"] = numpy.array(candi_rein, dtype="int32") - candi_rein[0]
            this_batch["target"] = numpy.array(this_target, dtype="int32")
            this_batch["result"] = numpy.array(this_result, dtype="int32")
            this_batch["zp_post"] = self.zp_post[zpi_s:zpi_e]
            this_batch["zp_pre"] = self.zp_pre[zpi_s:zpi_e]
            this_batch["zp_post_mask"] = self.zp_post_vec_mask[zpi_s:zpi_e]
            this_batch["zp_pre_mask"] = self.zp_pre_vec_mask[zpi_s:zpi_e]
            this_batch["candi"] = self.candi_vec[ci_s:ci_e]
            this_batch["candi_mask"] = self.candi_vec_mask[ci_s:ci_e]
            this_batch["np_post"] = self.np_post[ci_s:ci_e]
            this_batch["np_pre"] = self.np_pre[ci_s:ci_e]
            this_batch["np_post_mask"] = self.np_post_vec_mask[ci_s:ci_e]
            this_batch["np_pre_mask"] = self.np_pre_vec_mask[ci_s:ci_e]
            this_batch["fl"] = self.ifl_vec[ci_s:ci_e]
            # this_batch["infos"] = self.infos[ci_s:ci_e]
            this_batch["s2e"] = s2e
            correct_indexs, wrong_indexs, uni_correct_indexs, uni_wrong_indexs, correct_indexs1, correct_indexs2 = self.get_index(
                this_batch['s2e'], this_batch['result'])
            this_batch['cid'] = correct_indexs
            this_batch['wid'] = wrong_indexs
            this_batch['ucid'] = uni_correct_indexs
            this_batch['uwid'] = uni_wrong_indexs
            this_batch['cid1'] = correct_indexs1
            this_batch['cid2'] = correct_indexs2
            self.data_batch.append(this_batch)

    def get_index(self, s2e, result):
        correct_indexs, wrong_indexs, uni_correct_indexs, uni_wrong_indexs = [], [], [], []
        correct_indexs1, correct_indexs2 = [], []
        cnt = 0
        for s, e in s2e:
            if s == e:
                continue
            cur_label = result[s:e]
            correct_index = [i for i, ele in enumerate(cur_label) if ele == 1]
            wrong_index = [i for i, ele in enumerate(cur_label) if ele == 0]
            for cid in correct_index:
                for wid in wrong_index:
                    correct_indexs.append(cid + cnt)
                    wrong_indexs.append(wid + cnt)
            if len(wrong_index) == 0:
                for cid in correct_index:
                    uni_correct_indexs.append(cid + cnt)
            if len(correct_index) == 0:
                for wid in wrong_index:
                    uni_wrong_indexs.append(wid + cnt)
            for cid1 in correct_index:
                for cid2 in correct_index:
                    correct_indexs1.append(cid1 + cnt)
                    correct_indexs2.append(cid2 + cnt)
            cnt += e - s
        return correct_indexs, wrong_indexs, uni_correct_indexs, uni_wrong_indexs, correct_indexs1, correct_indexs2

    def devide(self, k=0.2):
        random.shuffle(self.data_batch)
        length = int(len(self.data_batch) * k)
        self.dev = self.data_batch[:length]
        self.train = self.data_batch[length:]
        self.data_batch = self.train

    def generate_data(self, shuffle=False):
        if shuffle:
            random.shuffle(self.data_batch)
        for data in self.data_batch:
            yield data

    def generate_dev_data(self, shuffle=False):
        if shuffle:
            random.shuffle(self.dev)
        for data in self.dev:
            yield data
