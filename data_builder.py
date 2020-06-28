# coding=utf8
import sys
import argparse
import cPickle
import random
import numpy
import timeit
import collections
import matplotlib.pyplot as plt
from buildTree import get_info_from_file
from data_generator import DataGenerator
from utils import *

DIR = "./data/"
parser = argparse.ArgumentParser(description="Experiments\n")
parser.add_argument("-data", default=DIR, type=str, help="saved vectorized data")
parser.add_argument("-raw_data", default="./data/zp_data/", type=str, help="raw_data")
parser.add_argument("-random_seed", default=0, type=int, help="random seed")
parser.add_argument("-batch_size", default=256, type=int, help="batch size")

args = parser.parse_args()
random.seed(0)
numpy.random.seed(0)


def get_sentence(zp_sentence_index, zp_index, nodes_info):
    nl, wl = nodes_info[zp_sentence_index]
    return_words = []
    for i in range(len(wl)):
        this_word = wl[i].word
        if i == zp_index:
            return_words.append("**pro**")
        else:
            if not (this_word == "*pro*"):
                return_words.append(this_word)
    return " ".join(return_words)


def get_candi_info(candi_sentence_index, nodes_info, candi_begin, candi_end, res_result):
    nl, wl = nodes_info[candi_sentence_index]
    candi_word = []
    for i in range(candi_begin, candi_end + 1):
        candi_word.append(wl[i].word)
    candi_word = "_".join(candi_word)

    candi_info = [str(res_result), candi_word]
    return candi_info


def list_vectorize(wl, words):
    il = []
    for w in wl:
        word = w.word
        if word in words:
            index = words.index(word)
        else:
            index = 0
        il.append(index)
    return il


def generate_vector(path, files):
    read_f = file('./data/emb', "rb")
    embedding, words, wd = cPickle.load(read_f)
    read_f.close()

    paths = [w.strip() for w in open(files).readlines()]

    total_sentence_num = 0
    vectorized_sentences = []
    zp_info = []

    startt = timeit.default_timer()
    is_test = True if 'test' in path else False
    for p in paths:
        if p.strip().endswith("DS_Store"):
            continue
        file_name = p.strip()
        if file_name.endswith('onf'):
            print 'Processing', file_name
            zps, azps, candi, nodes_info = get_info_from_file(file_name)

            anaphorics = []
            ana_zps = []
            for (zp_sentence_index, zp_begin_index, zp_end_index, antecedents, coref_id, is_real) in azps:
                for (candi_sentence_index, begin_word_index, end_word_index, coref_id) in antecedents:
                    anaphorics.append(
                        (zp_sentence_index, zp_begin_index, zp_end_index, candi_sentence_index, begin_word_index,
                         end_word_index))
                    ana_zps.append((zp_sentence_index, zp_begin_index, zp_end_index, is_real))

            si2reali = {}
            for k in nodes_info:
                nl, wl = nodes_info[k]
                vectorize_words = list_vectorize(wl, words)
                vectorized_sentences.append(vectorize_words)
                si2reali[k] = total_sentence_num
                total_sentence_num += 1

            for (sentence_index, zp_begin_index, zp_end_index, antecedents, coref_id, is_real) in azps:
                index_in_file = si2reali[sentence_index]
                zp = (index_in_file, sentence_index, zp_begin_index, zp_end_index)
                zp_nl, zp_wl = nodes_info[sentence_index]

                if (sentence_index, zp_begin_index, zp_end_index, is_real) not in ana_zps:
                    continue

                if is_test and is_real == 0:
                    continue

                candi_info = []
                for ci in range(max(0, sentence_index - 2), sentence_index + 1):
                    candi_sentence_index = ci
                    candi_nl, candi_wl = nodes_info[candi_sentence_index]

                    for (candi_begin, candi_end) in candi[candi_sentence_index]:
                        if ci == sentence_index and candi_end > zp_begin_index:
                            continue
                        res = 0
                        if (sentence_index, zp_begin_index, zp_end_index, candi_sentence_index, candi_begin,
                            candi_end) in anaphorics:
                            res = 1
                        candi_index_in_file = si2reali[candi_sentence_index]

                        ifl = get_fl((sentence_index, zp_begin_index, zp_end_index),
                                     (candi_sentence_index, candi_begin, candi_end),
                                     zp_wl, candi_wl, wd)

                        candidate = (
                            candi_index_in_file, candi_sentence_index, candi_begin, candi_end, res, -res, ifl)
                        candi_info.append(candidate)
                zp_info.append((zp, candi_info))

    endt = timeit.default_timer()
    print >> sys.stderr, "Total use %.3f seconds for Data Generating" % (endt - startt)
    vectorized_sentences = numpy.array(vectorized_sentences)
    return zp_info, vectorized_sentences


def generate_vector_data(test_only=False):
    DATA = args.raw_data
    train_data_path = args.data + "train/"
    test_data_path = args.data + "test/"

    if not test_only:
        train_zp_info, train_vectorized_sentences = generate_vector(DATA + "train/", "./data/train_list")
        train_vec_path = train_data_path + "sen.npy"
        numpy.save(train_vec_path, train_vectorized_sentences)
        save_f = file(train_data_path + "zp_info", 'wb')
        cPickle.dump(train_zp_info, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    test_zp_info, test_vectorized_sentences = generate_vector(DATA + "test/", "./data/test_list")
    test_vec_path = test_data_path + "sen.npy"
    numpy.save(test_vec_path, test_vectorized_sentences)
    save_f = file(test_data_path + "zp_info", 'wb')
    cPickle.dump(test_zp_info, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()


def generate_input_data(test_only=False):
    train_data_path = args.data + "train/"
    test_data_path = args.data + "test/"
    if not test_only:
        generate_vec(train_data_path)
    generate_vec(test_data_path)


def generate_vec(data_path):
    zp_candi_target = []
    zp_vec_index = 0
    candi_vec_index = 0

    zp_prefixs = []
    zp_prefixs_mask = []
    zp_postfixs = []
    zp_postfixs_mask = []
    candi_vecs = []
    candi_vecs_mask = []

    np_prefixs = []
    np_prefixs_mask = []
    np_postfixs = []
    np_postfixs_mask = []

    ifl_vecs = []

    infos = []

    read_f = file(data_path + "zp_info", "rb")
    zp_info_test = cPickle.load(read_f)
    read_f.close()

    vectorized_sentences = numpy.load(data_path + "sen.npy")
    for zp, candi_info in zp_info_test:
        index_in_file, sentence_index, zp_begin_index, zp_end_index = zp

        word_embedding_indexs = vectorized_sentences[index_in_file]
        max_index = len(word_embedding_indexs)

        prefix = word_embedding_indexs[max(0, zp_begin_index - 10):zp_begin_index]
        prefix_mask = (10 - len(prefix)) * [0] + len(prefix) * [1]
        prefix = (10 - len(prefix)) * [0] + prefix

        zp_prefixs.append(prefix)
        zp_prefixs_mask.append(prefix_mask)

        postfix = word_embedding_indexs[zp_end_index + 1:min(zp_end_index + 11, max_index)]
        postfix_mask = (len(postfix) * [1] + (10 - len(postfix)) * [0])[::-1]
        postfix = (postfix + (10 - len(postfix)) * [0])[::-1]

        zp_postfixs.append(postfix)
        zp_postfixs_mask.append(postfix_mask)

        candi_vec_index_inside = []
        for candi_index_in_file, candi_sentence_index, candi_begin, candi_end, res, target, ifl in candi_info:
            candi_word_embedding_indexs = vectorized_sentences[candi_index_in_file]
            candi_max_index = len(candi_word_embedding_indexs)

            candi_prefix = candi_word_embedding_indexs[max(0, candi_begin - 10):candi_begin]
            candi_prefix_mask = (10 - len(candi_prefix)) * [0] + len(candi_prefix) * [1]
            candi_prefix = (10 - len(candi_prefix)) * [0] + candi_prefix

            np_prefixs.append(candi_prefix)
            np_prefixs_mask.append(candi_prefix_mask)

            candi_postfix = candi_word_embedding_indexs[candi_end + 1:min(candi_end + 11, candi_max_index)]
            candi_postfix_mask = (len(candi_postfix) * [1] + (10 - len(candi_postfix)) * [0])[::-1]
            candi_postfix = (candi_postfix + (10 - len(candi_postfix)) * [0])[::-1]

            np_postfixs.append(candi_postfix)
            np_postfixs_mask.append(candi_postfix_mask)

            candi_vec = candi_word_embedding_indexs[candi_begin:candi_end + 1]
            if len(candi_vec) >= 8:
                candi_vec = candi_vec[-8:]
            candi_mask = (8 - len(candi_vec)) * [0] + len(candi_vec) * [1]
            candi_vec = (8 - len(candi_vec)) * [0] + candi_vec

            candi_vecs.append(candi_vec)
            candi_vecs_mask.append(candi_mask)

            ifl_vecs.append(ifl)
            infos.append(
                (index_in_file, sentence_index, zp_begin_index, zp_end_index, candi_index_in_file, candi_sentence_index,
                 candi_begin, candi_end))

            candi_vec_index_inside.append((candi_vec_index, res, target))

            candi_vec_index += 1

        zp_candi_target.append((zp_vec_index, candi_vec_index_inside))

        zp_vec_index += 1
    save_f = file(data_path + "zp_candi_pair_info", 'wb')
    cPickle.dump(zp_candi_target, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()

    zp_prefixs = numpy.array(zp_prefixs, dtype='int32')
    numpy.save(data_path + "zp_pre.npy", zp_prefixs)
    zp_prefixs_mask = numpy.array(zp_prefixs_mask, dtype='int32')
    numpy.save(data_path + "zp_pre_mask.npy", zp_prefixs_mask)
    zp_postfixs = numpy.array(zp_postfixs, dtype='int32')
    numpy.save(data_path + "zp_post.npy", zp_postfixs)
    zp_postfixs_mask = numpy.array(zp_postfixs_mask, dtype='int32')
    numpy.save(data_path + "zp_post_mask.npy", zp_postfixs_mask)
    candi_vecs = numpy.array(candi_vecs, dtype='int32')
    numpy.save(data_path + "candi_vec.npy", candi_vecs)
    candi_vecs_mask = numpy.array(candi_vecs_mask, dtype='int32')
    numpy.save(data_path + "candi_vec_mask.npy", candi_vecs_mask)

    np_prefixs = numpy.array(np_prefixs, dtype='int32')
    numpy.save(data_path + "np_pre.npy", np_prefixs)
    np_prefixs_mask = numpy.array(np_prefixs_mask, dtype='int32')
    numpy.save(data_path + "np_pre_mask.npy", np_prefixs_mask)
    np_postfixs = numpy.array(np_postfixs, dtype='int32')
    numpy.save(data_path + "np_post.npy", np_postfixs)
    np_postfixs_mask = numpy.array(np_postfixs_mask, dtype='int32')
    numpy.save(data_path + "np_post_mask.npy", np_postfixs_mask)

    assert len(ifl_vecs) == len(candi_vecs)

    ifl_vecs = numpy.array(ifl_vecs, dtype='float')
    numpy.save(data_path + "ifl_vec.npy", ifl_vecs)

    infos = numpy.array(infos, dtype='int32')
    numpy.save(data_path + "infos.npy", infos)


def get_head_verb(index, wl):
    father = wl[index].parent
    while father:
        leafs = father.get_leaf()
        for ln in leafs:
            if ln.tag.startswith("V"):
                return ln
        father = father.parent

    return None


def get_fl(zp, candidate, wl_zp, wl_candi, wd):
    ifl = []

    (zp_sentence_index, zp_begin_index, zp_end_index) = zp
    (candi_sentence_index, candi_index_begin, candi_index_end) = candidate

    sentence_dis = zp_sentence_index - candi_sentence_index

    # sentence distance
    tmp_ones = [0] * 3
    tmp_ones[sentence_dis] = 1
    ifl += tmp_ones

    cloNP = 0
    if sentence_dis == 0:
        if candi_index_end <= zp_begin_index:
            cloNP = 1
        for i in range(candi_index_end + 1, zp_begin_index):
            node = wl_zp[i]
            while True:
                if node.tag.startswith("NP"):
                    cloNP = 0
                    break
                node = node.parent
                if not node:
                    break
            if cloNP == 0:
                break

    tmp_ones = [0] * 2
    tmp_ones[cloNP] = 1
    ifl += tmp_ones

    first_zp = 1
    for i in range(zp_begin_index):
        if wl_zp[i].word == "*pro*":
            first_zp = 0
            break
    tmp_ones = [0] * 2
    tmp_ones[first_zp] = 1
    ifl += tmp_ones

    last_zp = 1
    for i in range(zp_end_index + 1, len(wl_zp)):
        if wl_zp[i].word == "*pro*":
            last_zp = 0
            break
    tmp_ones = [0] * 2
    tmp_ones[last_zp] = 1
    ifl += tmp_ones

    zp_node = wl_zp[zp_begin_index]
    NP_node = None
    father = zp_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    z_NP = 0
    if NP_node:
        z_NP = 1
    tmp_ones = [0] * 2
    tmp_ones[z_NP] = 1
    ifl += tmp_ones

    z_NinI = 0
    if NP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(NP_node):
                    z_NinI = 1
                break
            father = father.parent

    tmp_ones = [0] * 2
    tmp_ones[z_NinI] = 1
    ifl += tmp_ones

    VP_node = None
    zVP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            zVP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[zVP] = 1
    ifl += tmp_ones

    z_VinI = 0
    if VP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(VP_node):
                    z_VinI = 1
                break
            father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[z_VinI] = 1
    ifl += tmp_ones

    CP_node = None
    zCP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            zCP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[zCP] = 1
    ifl += tmp_ones

    tags = zp_node.parent.tag.split("-")
    zGram = 0
    zHl = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            zGram = 1
        if tags[1] == "HLN":
            zHl = 1
    tmp_ones = [0] * 2
    tmp_ones[zGram] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[zHl] = 1
    ifl += tmp_ones

    zc = 0
    if zCP == 1:
        zc = 1
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                zc = 2
                break
            if father == CP_node:
                break
            father = father.parent
    else:
        zc = 3
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent:  # 非根节点
                    zc = 4
                    break
            father = father.parent
    tmp_ones = [0] * 5
    tmp_ones[zc] = 1
    ifl += tmp_ones

    candi_node = wl_candi[candi_index_begin]
    NP_node = None
    father = candi_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    can_NinI = 0
    if NP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(NP_node):
                    can_NinI = 1
                break
            father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[can_NinI] = 1
    ifl += tmp_ones
    VP_node = None
    canVP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            canVP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[canVP] = 1
    ifl += tmp_ones
    can_VinI = 0
    if VP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(VP_node):
                    can_VinI = 1
                break
            father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[can_VinI] = 1
    ifl += tmp_ones
    CP_node = None
    canCP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            canCP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[canCP] = 1
    ifl += tmp_ones
    tags = candi_node.parent.tag.split("-")
    canGram = 0
    canADV = 0
    canTMP = 0
    canPN = 0
    canHl = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            canGram = 1
        elif tags[1] == "OBJ":
            canGram = 2
        if tags[1] == "ADV":
            canADV = 1
        if tags[1] == "TMP":
            canTMP = 1
        if tags[1] == "PN":
            canPN = 1
        if tags[1] == "HLN":
            canHl = 1
    tmp_ones = [0] * 3
    tmp_ones[canGram] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canADV] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canTMP] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canPN] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canHl] = 1
    ifl += tmp_ones
    canc = 0
    if canCP == 1:
        canc = 1
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                canc = 2
                break
            if father == CP_node:
                break
            father = father.parent
    else:
        canc = 3
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent:
                    canc = 4
                    break
            father = father.parent
    tmp_ones = [0] * 5
    tmp_ones[canc] = 1
    ifl += tmp_ones
    sibNV = 0
    if not sentence_dis == 0:
        sibNV = 0
    else:
        if abs(zp_begin_index - candi_index_end) == 1:
            sibNV = 1
        else:
            if abs(zp_begin_index - candi_index_begin) == 1:
                sibNV = 1
            else:
                if abs(zp_begin_index - candi_index_begin) == 2:
                    if zp_begin_index < candi_index_begin:
                        if wl_zp[zp_end_index + 1].tag == "PU":
                            sibNV = 1
                elif abs(zp_begin_index - candi_index_end) == 2:
                    if candi_index_end < zp_begin_index:
                        if wl_zp[zp_begin_index - 1].tag == "PU":
                            sibNV = 1
    tmp_ones = [0] * 2
    tmp_ones[sibNV] = 1
    ifl += tmp_ones
    gram_match = 0
    if not canGram == 0:
        if canGram == zGram:
            gram_match = 1
    tmp_ones = [0] * 2
    tmp_ones[gram_match] = 1
    ifl += tmp_ones

    chv = get_head_verb(candi_index_begin, wl_candi)
    zhv = get_head_verb(zp_begin_index, wl_zp)

    ch = wl_candi[candi_index_end]
    hc = "None"
    pc = "None"
    pz = "None"
    if ch:
        hc = ch.word
    if zhv:
        pz = zhv.word
    if chv:
        pc = chv.word
    tags = candi_node.parent.tag.split("-")
    canGram = "None"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            canGram = "SBJ"
        elif tags[1] == "OBJ":
            canGram = "OBJ"
    gc = canGram
    pcc = "None"
    for i in range(len(wl_zp) - 1, zp_end_index, -1):
        if wl_zp[i].tag.find("PU") >= 0:
            pcc = wl_zp[i].word
            break
    pc_pz = 0
    has = wd["%s_%s" % (hc, pcc)]
    if pc == pz:
        if canGram == "SBJ":
            pc_pz = 1
        elif canGram == "OBJ":
            pc_pz = 1
        else:
            pc_pz = 2
    tmp_ones = [0] * 3
    tmp_ones[pc_pz] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[has] = 1
    ifl += tmp_ones
    return ifl


# analysing data
# 1. dataset tag
# 2. Number of ZP, NP, Correct NP, Wrong NP
# 3. Number of NP, Correct NP, Wrong NP for every ZP
# 4. Distance of NP, Correct NP, Wrong NP for every ZP
def analysis_data(train_generator):
    test_generator = DataGenerator("test", 256)
    data = {'train': train_generator.generate_data(), 'dev': train_generator.generate_dev_data(),
            'test': test_generator.generate_data()}

    with open('results/data_analysis.txt', 'w') as f:
        for k, v in data.items():
            f.write('{}\n'.format(k))
            zp_num, np_num, correct_np_num, wrong_np_num = 0, 0, 0, 0
            np_num_l, correct_np_num_l, wrong_np_num_l = [], [], []
            np_dis_l, correct_np_dis_l, wrong_np_dis_l = [], [], []
            for d in v:
                zp_num += d['zp_pre'].shape[0]
                np_num += d['np_pre'].shape[0]
                correct_np_num += numpy.sum(d['result'])
                wrong_np_num += np_num - correct_np_num

                for s, e in d["s2e"]:
                    if s == e:
                        continue
                    np_num_tmp = e - s
                    correct_np_num_tmp = numpy.sum(d['result'][s:e])
                    wrong_np_num_tmp = np_num_tmp - correct_np_num_tmp

                    np_num_l.append(np_num_tmp)
                    correct_np_num_l.append(correct_np_num_tmp)
                    wrong_np_num_l.append(wrong_np_num_tmp)

                np_dis_l_tmp = numpy.dot(d['fl'][:, :3], [0, 1, 2])
                correct_np_dis_l_tmp = [j for i, j in zip(d['result'], np_dis_l_tmp) if i == 1]
                wrong_np_dis_l_tmp = [j for i, j in zip(d['result'], np_dis_l_tmp) if i == 0]
                np_dis_l.extend(np_dis_l_tmp)
                correct_np_dis_l.extend(correct_np_dis_l_tmp)
                wrong_np_dis_l.extend(wrong_np_dis_l_tmp)

            np_num_l = sorted(collections.Counter(np_num_l).items())
            correct_np_num_l = sorted(collections.Counter(correct_np_num_l).items())
            wrong_np_num_l = sorted(collections.Counter(wrong_np_num_l).items())

            np_dis_l = sorted(collections.Counter(np_dis_l).items())
            correct_np_dis_l = sorted(collections.Counter(correct_np_dis_l).items())
            wrong_np_dis_l = sorted(collections.Counter(wrong_np_dis_l).items())

            f.write('ZP Num: {}; NP Num: {}; Correct Num: {}; Wrong Num: {}\n'.format(zp_num, np_num, correct_np_num,
                                                                                      wrong_np_num))
            f.write('For ZPs:\nNP Num: {}\nCorrect NP Num: {}\nWrong NP Num: {}\n'.format(np_num_l, correct_np_num_l,
                                                                                          wrong_np_num_l))
            f.write('For NPs:\nNP Dis: {}\nCorrect NP Dis: {}\nWrong NP Dis: {}\n\n'.format(np_dis_l, correct_np_dis_l,
                                                                                            wrong_np_dis_l))

            plt.figure(figsize=(12.5, 7.5))

            plt.subplot(2, 3, 1)
            plt.bar([a for a, b in np_num_l], [b for a, b in np_num_l])
            plt.title('NP Num', fontsize=10)
            plt.subplot(2, 3, 2)
            plt.bar([a for a, b in correct_np_num_l], [b for a, b in correct_np_num_l])
            plt.title('Correct NP Num', fontsize=10)
            plt.subplot(2, 3, 3)
            plt.bar([a for a, b in wrong_np_num_l], [b for a, b in wrong_np_num_l])
            plt.title('Wrong NP Num', fontsize=10)

            plt.subplot(2, 3, 4)
            plt.bar([a for a, b in np_dis_l], [b for a, b in np_dis_l])
            plt.title('NP Dis', fontsize=10)
            plt.subplot(2, 3, 5)
            plt.bar([a for a, b in correct_np_dis_l], [b for a, b in correct_np_dis_l])
            plt.title('Correct NP Dis', fontsize=10)
            plt.subplot(2, 3, 6)
            plt.bar([a for a, b in wrong_np_dis_l], [b for a, b in wrong_np_dis_l])
            plt.title('Wrong NP Dis', fontsize=10)

            plt.savefig('results/data_analysis.png')


if __name__ == '__main__':
    # build data from raw OntoNotes data
    print 'Processing'
    generate_vector_data()
    generate_input_data()
    # split training data into dev and train, saved in ./data/train_data
    print 'Dividing'
    train_generator = DataGenerator("train", args.batch_size)
    train_generator.devide()
    save_f = file("./data/train_data", 'wb')
    cPickle.dump(train_generator, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()
    print 'Analysing'
    analysis_data(train_generator)
