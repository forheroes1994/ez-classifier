# 提取文件中的所有单词和标签
from instence import *
class Read_inst:
    def read(self, path):
        sentence = []
        # f = open('D:\\classifier\\data\\raw.clean.dev', encoding="UTF-8")
        f = open(path, encoding="UTF-8")
        for line in f.readlines():
            m_1 = inst()
            x = line.strip().split('|||')
            m_1.m_word = x[0].strip().split(' ')
            m_1.m_label = x[1].strip()
            sentence.append(m_1)
        f.close()
        return sentence