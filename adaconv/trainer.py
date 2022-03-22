import os
import datetime
import tensorflow as tf
from contextlib import nullcontext
from dataload import DataLoad
from model import StyleTransfer

class Trainer(DataLoad):
    """
    Model trainer class
    """
    def __init__(self):
        super(Trainer, self).__init__()
        
        self.gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for _device in self.gpu_devices:
            tf.config.experimental.set_memory_growth(_device, True)
            print(f"TF using {_device}")
        tf.config.set_soft_device_placement(True)
        if len(self.gpu_devices):
            self.use_mixed_precision = False
            self.device = "/gpu:0"
        else:
            self.use_mixed_precision = False
            self.device = "/cpu:0"

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

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
        self.batch_size = 8
        self.style_weight = 1.0

        # Iteration
        self.num_iteration = 100000

        # Logging
        self.log_step = 10
        self.save_step = 1000
        self.summary_step = 100
        self.freeze_encoder = True

    def build(self):
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        exp_decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(exp_decay_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        if self.use_mixed_precision:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer, dynamic=True, initial_scale=2**4)
        
        self.mse = tf.keras.losses.MeanSquaredError()
        
        self.styletransfer = StyleTransfer(image_size = self.image_size, 
                                           in_channels = self.dec_in_channels,
                                           out_channels = self.dec_out_channels,
                                           n_groups = self.dec_n_groups,
                                           n_convs = self.dec_n_convs,
                                           Sd = self.Sd,
                                           Sh = self.Sh,
                                           Sw = self.Sw,
                                           mean = self.mean,
                                           freeze_encoder = self.freeze_encoder, 
                                           name = "StyleTransfer")

        self.checkpoint = tf.train.Checkpoint(step = self.global_step,
                                              optimizer = self.optimizer,
                                              styletransfer = self.styletransfer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.logdir,"ckpts"),
                                                             checkpoint_name='ckpt',
                                                             max_to_keep=3)

        self.built = True
    
    def mse_content_loss(self, content_feat : tf.Tensor, output_feat : tf.Tensor) -> tf.Tensor:
        content_feat = tf.cast(content_feat, dtype = tf.float32)
        output_feat = tf.cast(output_feat, dtype = tf.float32)
        return self.mse(content_feat, output_feat)

    def moments_match_style_loss(self, style_feats : list[tf.Tensor], output_feats : list[tf.Tensor]) -> tf.Tensor:
        
        _mean_loss_list = list()
        _std_loss_list = list()
        for s_feat, x_feat in zip(style_feats, output_feats):
            s_feat = tf.cast(s_feat, dtype = tf.float32)
            x_feat = tf.cast(x_feat, dtype = tf.float32)
            s_mean, s_var = tf.nn.moments(s_feat, [1,2], keepdims=True)
            x_mean, x_var = tf.nn.moments(x_feat, [1,2], keepdims=True)
            s_std = tf.math.sqrt(s_var + 1e-3)
            x_std = tf.math.sqrt(x_var + 1e-3)
            mean_loss = self.mse(s_mean, x_mean)
            std_loss = self.mse(s_std, x_std)
            _mean_loss_list.append(mean_loss)
            _std_loss_list.append(std_loss)
        
        tot_mean_loss = tf.math.add_n(_mean_loss_list)
        tot_std_loss = tf.math.add_n(_std_loss_list)
        
        return tot_mean_loss + tot_std_loss
    
    def load_ckpts(self, partial : bool = False):

        if partial:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
        else:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("No checkpoints, Initializing from scratch.")
    
    def train(self):

        log_dir = os.path.join(self.logdir, "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)

        writer = tf.summary.create_file_writer(log_dir)

        self.load_ckpts()
        self.get_dataset()
        content_train_dataloader = self.create_dataloader(self.content_train_list).__iter__()
        content_test_dataloader = self.create_dataloader(self.content_test_list).__iter__()
        style_train_dataloader = self.create_dataloader(self.style_train_list).__iter__()
        style_test_dataloader = self.create_dataloader(self.style_test_list).__iter__()

        with writer.as_default():
            with tf.summary.record_if(True):
                while int(self.checkpoint.step) <= self.num_iteration:
                    batch_contents = next(content_train_dataloader)
                    batch_styles = next(style_train_dataloader)

                    # Train Step
                    start_time = datetime.datetime.now()
                    styled_content, loss, style_loss, content_loss = self.train_step(batch_contents, batch_styles, training = True)
                    duration = datetime.datetime.now() - start_time
                    
                    if int(self.checkpoint.step) % self.summary_step == 0:
                        test_batch_contents = next(content_test_dataloader)
                        test_batch_styles = next(style_test_dataloader)
                        test_styled_content, test_loss, test_style_loss, test_content_loss = self.train_step(test_batch_contents, test_batch_styles, training = False)
                        
                        # Write Summary
                        self.write_summary(loss, style_loss, content_loss, batch_contents, batch_styles, styled_content, prefix = 'train')
                        self.write_summary(test_loss, test_style_loss, test_content_loss, test_batch_contents, test_batch_styles, test_styled_content, prefix = 'test')

                    self.checkpoint.step.assign_add(1)

                    if int(self.checkpoint.step) % self.save_step == 0:
                        _ = self.checkpoint_manager.save(checkpoint_number = int(self.checkpoint.step))
                    
                    if int(self.checkpoint.step) % self.log_step == 0:
                        print(f"{datetime.datetime.now()} step {int(self.checkpoint.step)}, loss = {loss}, style_loss = {style_loss}, content_loss = {content_loss}, ({(self.batch_size/duration.total_seconds())} examples/sec; {duration.total_seconds()} sec/batch)")

        print("Training Done.")

    def write_summary(self, loss : tf.Tensor, style_loss : tf.Tensor, content_loss : tf.Tensor, contents : tf.Tensor, styles : tf.Tensor, styled_content : tf.Tensor, prefix : str = ''):
        tf.summary.scalar(prefix + "_loss", loss, step=int(self.checkpoint.step))
        tf.summary.scalar(prefix + "_style_loss", style_loss, step=int(self.checkpoint.step))
        tf.summary.scalar(prefix + "_content_loss", content_loss, step=int(self.checkpoint.step))
        
        tf.summary.image(prefix + "_content_images", contents, step=int(self.checkpoint.step))
        tf.summary.image(prefix + "_style_images", styles, step=int(self.checkpoint.step))
        tf.summary.image(prefix + "_styled_content_images", styled_content, step=int(self.checkpoint.step))
    
    @tf.function
    def train_step(self, contents : tf.Tensor, styles : tf.Tensor, training : bool = True) -> tuple[tf.Tensor]:
        with tf.device(self.device):
            cnxtmngr = tf.GradientTape() if training else nullcontext()
            with cnxtmngr as tape:
                styled_content, feats = self.styletransfer(inputs = [contents, styles], training = training, compute_feats = True)
                C_feats, S_feats, x_feats = feats

                content_loss = self.mse_content_loss(C_feats[-1], x_feats[-1]) * 2.5e-8
                style_loss = self.moments_match_style_loss(S_feats, x_feats) * 1e-6

                loss = content_loss + style_loss * self.style_weight
                if self.use_mixed_precision:
                    loss = self.optimizer.get_scaled_loss(loss)
                
                trainable_variables = self.styletransfer.trainable_variables
            
            if training:
                grads = tape.gradient(loss, trainable_variables)
                if self.use_mixed_precision:
                    grads = self.optimizer.get_unscaled_gradients(grads)
                grads = [(tf.clip_by_value(grad, -1, 1)) for grad in grads]

                self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        return styled_content, loss, style_loss, content_loss

    def update(self, newdata : dict):
        for key,value in newdata.items():
            setattr(self,key,value)