import os

class hyperparameters():

    def __init__(self):
        # Model
        # self.mean = [103.939, 116.779, 123.68] # "caffe mode of preprocessing input"
        self.model_type = "AdaConv" # "AdaConv", "AdaIn"
        self.freeze_encoder = True
        self.vgg19_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
        self.dec_in_channels = [512, 256, 128, 64]
        self.dec_out_channels = [256, 128, 64, 3]
        self.dec_n_convs = [1, 4, 2, 2]
        if self.model_type == "AdaConv":
            self.dec_n_groups = [512, 256//2, 128//4, 64//8] # -> [Cin,Cin//2,Cin//4,Cin//8] -> [512, 128, 32, 8] 
            self.Sd = 512
            self.Sh = 3
            self.Sw = 3

        # Dataset
        self.image_size = 256
        self.random_crop_prob = 0.2
        self.resize_size = self.image_size * 2
        
        # Training
        self.learning_rate = 1e-4
        self.batch_size = 8 if self.model_type == "AdaConv" else 16
        self.style_weight = 10.0

        # Iteration
        self.num_iteration = 100000

        # Logging
        self.log_step = 10
        self.save_step = 1000
        self.summary_step = 100
        self.freeze_encoder = True

        # Dataset
        self.raw_data = os.path.join("../data", 'raw')

        # logdir
        self.logdir = os.path.join("../results/models", self.model_type)

    def update(self, newdata : dict):
        for key,value in newdata.items():
            setattr(self,key,value)

Hyperparams = hyperparameters()