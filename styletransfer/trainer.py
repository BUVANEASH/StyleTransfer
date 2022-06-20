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
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        
        self.gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for _device in self.gpu_devices:
            tf.config.experimental.set_memory_growth(_device, True)
            print(f"TF using {_device}")
        tf.config.set_soft_device_placement(True)
        if len(self.gpu_devices):
            self.use_mixed_precision = True
            self.device = "/gpu:0"
        else:
            self.use_mixed_precision = False
            self.device = "/cpu:0"

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.update(kwargs)

    def build(self):
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        exp_decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, 
                                                                      decay_steps=100000, 
                                                                      decay_rate=0.96, 
                                                                      staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(exp_decay_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        if self.use_mixed_precision:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer, dynamic=True, initial_scale=2**4)
        
        self.mse = tf.keras.losses.MeanSquaredError()

        kwargs = {"model_type" : self.model_type,
                  "in_channels" : self.dec_in_channels,
                  "out_channels" : self.dec_out_channels,
                  "n_convs" : self.dec_n_convs,
                  # "mean" : self.mean,
                  "image_size" : self.image_size,
                  "freeze_encoder" : self.freeze_encoder,
                  "backbone_output_layers" : self.vgg19_layers}
        
        if self.model_type == "AdaConv":
            kwargs.update({"n_groups" : self.dec_n_groups,
                           "Sd" : self.Sd,
                           "Sh" : self.Sh,
                           "Sw" : self.Sw})
        
        kwargs.update({"name" : f"{self.model_type}_StyleTransfer"})

        self.styletransfer = StyleTransfer(**kwargs)

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
                    styled_content, loss, \
                        style_loss, content_loss = self.train_step(contents = batch_contents, 
                                                                   styles = batch_styles, 
                                                                   training = True)
                    duration = datetime.datetime.now() - start_time
                    
                    if int(self.checkpoint.step) % self.summary_step == 0:
                        test_batch_contents = next(content_test_dataloader)
                        test_batch_styles = next(style_test_dataloader)
                        test_styled_content, test_loss, \
                            test_style_loss, test_content_loss = self.train_step(contents = test_batch_contents, 
                                                                                 styles = test_batch_styles, 
                                                                                 training = False)
                        
                        # Write Summary
                        self.write_summary(loss = loss, 
                                           style_loss = style_loss, 
                                           content_loss = content_loss, 
                                           contents = batch_contents, 
                                           styles = batch_styles, 
                                           styled_content = styled_content, 
                                           prefix = 'train')
                        
                        self.write_summary(loss = test_loss, 
                                           style_loss = test_style_loss, 
                                           content_loss = test_content_loss, 
                                           contents = test_batch_contents, 
                                           styles = test_batch_styles, 
                                           styled_content = test_styled_content, 
                                           prefix = 'test')

                    self.checkpoint.step.assign_add(1)

                    if int(self.checkpoint.step) % self.save_step == 0:
                        _ = self.checkpoint_manager.save(checkpoint_number = int(self.checkpoint.step))
                    
                    if int(self.checkpoint.step) % self.log_step == 0:
                        _log_str = "{0} step {1}, loss = {2:.4f}, style_loss = {3:.4f}, content_loss = {4:.4f}, ({5:.4f} examples/sec; {6:.4f} sec/batch)"
                        print(_log_str.format(datetime.datetime.now(),
                                              int(self.checkpoint.step),
                                              loss,
                                              style_loss,
                                              content_loss,
                                              (self.batch_size/duration.total_seconds()),
                                              duration.total_seconds()))

        print("Training Done.")

    def write_summary(self, 
                      loss : tf.Tensor, 
                      style_loss : tf.Tensor, 
                      content_loss : tf.Tensor, 
                      contents : tf.Tensor, 
                      styles : tf.Tensor, 
                      styled_content : tf.Tensor, 
                      prefix : str = ''):

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
                styled_content, feats = self.styletransfer(inputs = [contents, styles], 
                                                           training = training, 
                                                           get_feats = True)
                
                x_feats = feats.get("combined_features")
                S_feats = feats.get("style_features")
                C_feats = feats.get("content_features")

                content_loss = self.mse_content_loss(content_feat = C_feats[-1], output_feat = x_feats[-1])
                style_loss = self.moments_match_style_loss(style_feats = S_feats, output_feats = x_feats)

                loss = content_loss + (style_loss * self.style_weight)
                if self.use_mixed_precision:
                    _loss = self.optimizer.get_scaled_loss(loss)
                else:
                    _loss = loss
                
                trainable_variables = self.styletransfer.trainable_variables
            
            if training:
                grads = tape.gradient(_loss, trainable_variables)
                if self.use_mixed_precision:
                    grads = self.optimizer.get_unscaled_gradients(grads)

                if self.grad_clip["type"] == "clip_by_value":
                    grads = [(tf.clip_by_value(grad, self.grad_clip["value"][0], self.grad_clip["value"][1])) for grad in grads]
                elif self.grad_clip["type"] == "clip_by_norm":
                    grads = [(tf.clip_by_norm(grad, self.grad_clip["value"])) for grad in grads]

                self.optimizer.apply_gradients(zip(grads, trainable_variables))

                if int(self.checkpoint.step) % self.log_step == 0:
                    _step = tf.cast(self.checkpoint.step, dtype = tf.int64)
                    for g, v in zip(grads, trainable_variables):
                        tf.summary.histogram(f"grads/{str(v.name)}", g, step = _step)
                        tf.summary.histogram(f"weights/{str(v.name)}", v, step = _step)
        
        return styled_content, loss, style_loss, content_loss

    def update(self, newdata : dict):
        for key,value in newdata.items():
            setattr(self,key,value)