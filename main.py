import os
import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from hyperparameter import *
from model import CNN_Text
from read_file import Read_inst
from instence import *
import random

torch.manual_seed(100)
random.seed(100)


class Classifier:
    def __init__(self):
        self.hyperparameter_1 = Hyperparameter()
        self.inst = inst()
        self.Read_inst = Read_inst()
        self.aphabet = alphabet()
        self.example = example()

    def divide_two_dict(self, m_2):
        all_w = []
        all_l = []
        for inst in m_2:
            for w in inst.m_word:
                all_w.append(w)
        all_w.append(self.hyperparameter_1.unknow)
        all_w.append(self.hyperparameter_1.padding)
        word_alphabet = self.aphabet.add_dict(all_w)
        for inst in m_2:
            for w in inst.m_label:
                all_l.append(w)
        label_alphabet = self.aphabet.add_dict(all_l)
        return word_alphabet, label_alphabet

    def load_my_vector(self, path, vocab):
        word_vecs = {}
        with open(path, encoding= 'UTF - 8') as f:
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(' ')
                word = values[0]
                if word in vocab:
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknow_words_by_uniform(self, word_vecs, vocab, k = 100):
        list_word2vec = []
        oov = 0
        iov = 0
        for word in vocab:
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        return list_word2vec

    def add_unknow_words_by_average(self, word_vecs, vocab, k = 100 ):
        word_vecs_numpy = []
        for word in vocab:
            if word in word_vecs:
                word_vecs_numpy.append(word_vecs[word])
        col = []
        for i in range(k):
            sum = 0.0
            for j in range(int(len(word_vecs_numpy))):
                sum += word_vecs_numpy[j][i]
                sum = round(sum, 6)
            col.append(sum)
        zero = []
        for m in range(k):
            avg = col[m] / (len(word_vecs_numpy))
            avg = round(avg, 6)
            zero.append(float(avg))
        list_word2vec = []
        oov = 0
        iov = 0
        for word in vocab:
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        return list_word2vec


    def get_max_sentence_len(self, all_example):
        max_sentence_len = 0
        for exam in all_example:
            if max_sentence_len < len(exam.m_word_index):
                max_sentence_len = len(exam.m_word_index)
        return max_sentence_len

    def batch(self, examples, batch_size, max_len):
        for exam in examples:
            if len(exam.m_word_index) == max_len:
                continue
            for i in range(max_len - len(exam.m_word_index)):
                exam.m_word_index.append(self.hyperparameter_1.padding_id)
        minibatch_word = []
        minibatch_label = []
        for exam in examples:
            minibatch_word.append(exam.m_word_index)
            minibatch_label.append(exam.m_label_index)

            if len(minibatch_word) % batch_size == 0:
                minibatch_word = Variable(torch.LongTensor(minibatch_word))
                minibatch_label = Variable(torch.LongTensor(minibatch_label))
                return minibatch_word, minibatch_label
        if minibatch_word or minibatch_label:
            minibatch_word = Variable(torch.LongTensor(minibatch_word))
            minibatch_label = Variable(torch.LongTensor(minibatch_label))
            return minibatch_word, minibatch_label

    def set_batchBlock(self, examples):
        if len(examples) % self.hyperparameter_1.batch_size == 0:
            batchBlock = len(examples) // self.hyperparameter_1.batch_size
        else:
            batchBlock = len(examples) // self.hyperparameter_1.batch_size + 1
        return batchBlock

    def set_index(self, examples):
        index = []
        for i in range(len(examples)):
            index.append(i)
        return index

    def out_example_index(self, m_2, m_3):
        word_dict, label_dict = self.divide_two_dict(m_2)
        all_example = []
        for i in m_3:
            b = example()
            b.m_label_index.append(label_dict.dict[i.m_label])
            for j in i.m_word:
                if j not in word_dict.dict:
                    b.m_word_index.append(word_dict.dict[self.hyperparameter_1.unknow])
                else:
                    b.m_word_index.append(word_dict.dict[j])
            all_example.append(b)
        self.hyperparameter_1.unknow_id = word_dict.dict[self.hyperparameter_1.unknow]
        self.hyperparameter_1.padding_id = word_dict.dict[self.hyperparameter_1.padding]
        self.hyperparameter_1.vocab_num = len(word_dict.m_list)
        return all_example

    def train(self, m_2,m_3,m_4):
        word_dict, label_dict = self.divide_two_dict(m_2)
        if self.hyperparameter_1.word_embedding:
            path = "word2vec/glove.6B.100d.txt"
            print("loading word2vec ")
            word_vecs = self.load_my_vector(path, word_dict.m_list)
            print("new words already in word2vec:" + str(len(word_vecs)))
            print("loading unknow word2vec and convert to list... ")
            word_vecs = self.add_unknow_words_by_average(word_vecs, word_dict.m_list, k=self.hyperparameter_1.embed_dim)
            print("unknown word2vec load ! and converted to list...")
        # if self.hyperparameter_1.word_embedding:
            self.hyperparameter_1.pretrained_weight = word_vecs
            # pretrained_weight = np.array(self.hyperparameter_1.pretrained_weight)
            # self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))


        # self.nn = network(2, 2, 2, hidden_layer_weights=None, hidden_layer_bias=None, output_layer_weights=None, output_layer_bias=None)
        train_example = self.out_example_index(m_2,m_2)
        dev_example = self.out_example_index(m_2, m_3)
        test_example = self.out_example_index(m_2, m_4)

        random.shuffle(train_example)
        random.shuffle(dev_example)
        random.shuffle(test_example)


        self.model = CNN_Text(self.hyperparameter_1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameter_1.lr)
        train_example_idx = self.set_index(train_example)
        random.shuffle(train_example_idx)
        steps = 0
        self.model.train()
        for epoch in range(1, self.hyperparameter_1.epochs + 1):
            batchBlock = self.set_batchBlock(train_example)
            for every_batchBlock in range(batchBlock):
                exams = []
                start_pos = every_batchBlock * self.hyperparameter_1.batch_size
                end_pos = (every_batchBlock + 1) * self.hyperparameter_1.batch_size
                if end_pos > len(train_example):
                    end_pos = len(train_example)
                for idx in range(start_pos, end_pos):
                    exams.append(train_example[train_example_idx[idx]])
                max_len = self.get_max_sentence_len(exams)
                optimizer.zero_grad()
                feat, label = self.batch(exams, self.hyperparameter_1.batch_size, max_len)
                label = label.view(len(exams))
                logit = self.model.forward(feat)
                loss = F.cross_entropy(logit, label)
                loss.backward()
                optimizer.step()
                steps += 1
                if steps % self.hyperparameter_1.log_interval == 0:
                    train_size = len(train_example)
                    corrects = (torch.max(logit, 1)[1].view(label.size()).data == label.data).sum()
                    accuracy = corrects / self.hyperparameter_1.batch_size * 100.0
                    sys.stdout.write(
                        '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                    train_size,
                                                                                    loss.data[0],
                                                                                    accuracy,
                                                                                    corrects,
                                                                                    self.hyperparameter_1.batch_size))
                if steps % self.hyperparameter_1.test_interval == 0:
                    self.eval(dev_example, self.model)
                if steps % self.hyperparameter_1.save_interval == 0:
                    if not os.path.isdir(self.hyperparameter_1.save_dir): os.makedirs(self.hyperparameter_1.save_dir)
                    save_prefix = os.path.join(self.hyperparameter_1.save_dir, 'snapshot')
                    save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                    torch.save(self.model, save_path)

    def eval(self, data_example, model):
        self.model.eval()
        corrects, avg_loss = 0, 0
        data_example_idx = self.set_index(data_example)
        batchBlock = self.set_batchBlock(data_example)
        for every_batchBlock in range(batchBlock):
            exams = []
            start_pos = every_batchBlock * self.hyperparameter_1.batch_size
            end_pos = (every_batchBlock + 1) * self.hyperparameter_1.batch_size
            if end_pos > len(data_example):
                end_pos = len(data_example)
            for idx in range(start_pos, end_pos):
                exams.append(data_example[data_example_idx[idx]])
            max_len = self.get_max_sentence_len(exams)
            feat, label = self.batch(exams, self.hyperparameter_1.batch_size, max_len)
            label = label.view(len(exams))
            logit = self.model.forward(feat)
            loss = F.cross_entropy(logit, label, size_average=False)
            # print(loss.data[0])
            avg_loss += loss.data[0]
            corrects += (torch.max(logit, 1)
                            [1].view(label.size()).data == label.data).sum()

        size = len(data_example)
        avg_loss = loss.data[0] / size
        accuracy = corrects / size * 100.0
        self.model.train()
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                        accuracy,
                                                                        corrects,
                                                                        size))

    def variable(self, example):
        x = Variable(torch.LongTensor(1, len(example.m_word_index)))
        y = Variable(torch.LongTensor(1))
        for i in range(len(example.m_word_index)):
            x.data[0][i] = example.m_word_index[i]
        y.data[0] = example.m_label_index[0]
        return x, y


a = Classifier()
b = Read_inst()
m_2 = b.read("E:\\classifier\\data\\raw.clean.train")
m_3 = b.read("E:\\classifier\\data\\raw.clean.dev")
m_4 = b.read("E:\\classifier\\data\\raw.clean.test")
a.train(m_2, m_3, m_4)





























