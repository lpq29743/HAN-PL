# coding=utf8
from data_generator import *
from net import *
from conf import *
import sys
import numpy

random.seed(0)
numpy.random.seed(0)
sys.setrecursionlimit(1000000)





# 获取文件中句子
def get_info_from_file(file_name):
    f = open(file_name)
    cut_sentences = []
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if line == 'Treebanked sentence:':
            f.readline()
            sentence = []
            line = f.readline().strip()
            while line != '':
                sentence.extend(line.split(' '))
                line = f.readline().strip()
            cut_sentences.append(sentence)
    return cut_sentences


# 句子转换为对应 id
def list_vectorized(wl, words):
    il = []
    for word in wl:
        if word in words:
            index = words.index(word)
        else:
            index = 0
        il.append(index)
    return il


# 向量化句子
def vectorized_sentences(file_name):
    sentences = []
    read_f = file('./data/emb', 'rb')
    embedding, words, wd = cPickle.load(read_f)
    read_f.close()

    cut_sentences = get_info_from_file(file_name)
    for cut_sentence in cut_sentences:
        vectorized_words = list_vectorized(cut_sentence, words)
        sentences.append((' '.join(cut_sentence), vectorized_words))
    return sentences


# 获取原数据内容
def get_raw_data(arg):
    paths = [w.strip() for w in open(arg).readlines()]
    sentences = []
    for p in paths:
        sentences.extend(vectorized_sentences(p.strip()))
    return sentences


# 获取预处理后内容
def get_processed_data(data_path):
    processed_data = []

    read_f = file(data_path + 'zp_info', 'rb')
    zp_info_test = cPickle.load(read_f)
    read_f.close()

    vectorized_sentences = numpy.load(data_path + 'sen.npy')
    for zp, candi_info in zp_info_test:
        index_in_file, sentence_index, zp_begin_index, zp_end_index = zp
        word_embedding_indexs = vectorized_sentences[index_in_file]
        max_index = len(word_embedding_indexs)

        prefix = word_embedding_indexs[max(0, zp_begin_index - 10):zp_begin_index]

        postfix = word_embedding_indexs[zp_end_index + 1:min(zp_end_index + 11, max_index)]

        candis = []
        for candi_index_in_file, candi_sentence_index, candi_begin, candi_end, res, target, ifl in candi_info:
            candi_word_embedding_indexs = vectorized_sentences[candi_index_in_file]
            candi_vec = candi_word_embedding_indexs[candi_begin:candi_end + 1]
            candis.append((candi_index_in_file, candi_begin, candi_end, candi_vec))
        processed_data.append((index_in_file, zp_begin_index, zp_end_index, prefix, postfix, candis))

    return processed_data


# 获取预测结果
def get_predict_max(data):
    predicts = []
    predict_ids = []
    wrong_res = []
    for id, (result, output) in enumerate(data):
        if isinstance(result, int) and isinstance(output, int):
            predicts.append(-1)
            predict_ids.append(-1)
            continue
        max_index = -1
        max_pro = 0.0
        for i in range(len(output)):
            if output[i] > max_pro:
                max_index = i
                max_pro = output[i]
        if result[max_index] != 1:
            correct_id, correct_score = 0, -1
            for i in range(len(output)):
                if result[i] == 1:
                    correct_id = i
                    correct_score = output[i]
            wrong_res.append((id, max_index, max_pro, correct_id, correct_score))
        predicts.append(result[max_index])
        predict_ids.append(max_index)
    return predicts, predict_ids, wrong_res


# 获取模型消解结果
def get_result(model_name, generator):
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.set_device(args.gpu)

    read_f = file('./data/emb', 'rb')
    embedding_matrix, _, _ = cPickle.load(read_f)
    read_f.close()

    print 'Building torch model'
    best_model = torch.load('./models/' + model_name)

    predict = []
    for data in generator.generate_data():
        zp_rep, npc_rep, np_rep, feature = best_model.forward(data)
        output = best_model.generate_score(zp_rep, npc_rep, np_rep, feature)
        output = output.data.cpu().numpy()
        for s, e in data["s2e"]:
            if s == e:
                predict.append((-1, -1))
                continue
            predict.append((data['result'][s:e], output[s:e]))

    predicts, predict_id, wrong_res = get_predict_max(predict)

    save_f = file(model_name + '_predict', 'wb')
    cPickle.dump(predicts, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    return wrong_res


# 可视化结果
def vis_result(processed_data, sentences, info):
    search_id, wrong_id, wrong_sco, correct_id, correct_sco = info
    index_in_file, zp_begin_index, zp_end_index, zp_prefixs, zp_postfixs, candis = processed_data[search_id]
    wrong_candi_index_in_file, wrong_candi_begin, wrong_candi_end, _ = candis[wrong_id]
    correct_candi_index_in_file, correct_candi_begin, correct_candi_end, _ = candis[correct_id]

    return '错误 ZP 所在句子：%s\n位置：%s\n错的先行词所在句子：%s\n位置：%s-%s，分数：%s\n对的先行词所在句子：%s\n位置：%s-%s，分数：%s\n' % (
        sentences[index_in_file][0], zp_begin_index, sentences[wrong_candi_index_in_file][0], wrong_candi_begin,
        wrong_candi_end, wrong_sco, sentences[correct_candi_index_in_file][0], correct_candi_begin, correct_candi_end,
        correct_sco)


if __name__ == '__main__':
    # 查看处理数据情况
    # vis_raw_data()

    # 可视化实验结果
    is_test = False
    if is_test:
        raw_data_file = './data/test_list'
        processed_data_file = './data/test/'
        generator = DataGenerator("test", 256)
    else:
        raw_data_file = './data/train_list'
        processed_data_file = './data/train/'
        generator = DataGenerator('train', 256)

    # get raw data
    sentences = get_raw_data(raw_data_file)
    # get processed data
    processed_data = get_processed_data(processed_data_file)
    # get result
    wrong_res = get_result('model', generator)
    # visualize result
    with open('analysis.txt', 'w') as f:
        for info in wrong_res:
            analysis_res = vis_result(processed_data, sentences, info)
            f.write(analysis_res)
