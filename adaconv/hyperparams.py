import os

class hyperparameters():

    def __init__(self):
        # Model
        self.Sh = 3
        self.Sw = 3
        self.Sd = 512
        self.dec_n_convs = [1, 4, 2, 2]
        self.dec_in_channels = [512, 256, 128, 64]
        self.dec_out_channels = [256, 128, 64, 3]
        self.dec_n_groups = [512, 256//2, 128//4, 64//8] # [512, 128, 32, 8]
        self.mean = [103.939, 116.779, 123.68] # "caffe mode of preprocessing input"

        # Dataset
        self.image_size = 256
        self.random_crop_prob = 0.5
        self.resize_size = self.image_size * 2
        
        # Training
        self.learning_rate = 1e-4
        self.batch_size = 4
        self.style_weight = 10.0

        # Iteration
        self.num_iteration = 200000

        # Logging
        self.log_step = 10
        self.save_step = 1000
        self.summary_step = 100
        self.freeze_encoder = True

        # Dataset
        self.data_path = "../data"
        self.raw_data = os.path.join(self.data_path,'raw')

        # logdir
        self.logdir = "../results/models"

    def update(self, newdata : dict):
        for key,value in newdata.items():
            setattr(self,key,value)

Hyperparams = hyperparameters()