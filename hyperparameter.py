class Hyperparameter:
    def __init__(self):


        self.lr = 0.001
        self.unknow = "---unknow---"
        self.unknow_id = 0
        self.padding = "---padding---"
        self.padding_id = 0

        self.embed_dim = 100
        self.vocab_num = 0
        self.class_num = 5
        self.batch_size = 16
        self.log_interval = 1
        self.epochs = 256
        self.test_interval = 100
        self.save_interval = 100
        self.save_dir = 'snapshot'
        self.word_embedding = False





