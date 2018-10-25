from collections import OrderedDict
class inst:
    def __init__(self):
        self.m_word = []
        self.m_label = ''


class alphabet:
    def __init__(self):
        self.m_list = []
        self.dict = OrderedDict()

    def add_dict(self, elem):
        a = alphabet()
        for e in elem:
            if e not in a.m_list:
                a.m_list.append(e)
        for idx in range(len(a.m_list)):
            e = a.m_list[idx]
            a.dict[e] = idx
        return a



class example:
    def __init__(self):
        self.m_word_index = []
        self.m_label_index = []
        self.feat = Feature()

class Feature:
    def __init__(self):
        self.length = 0

