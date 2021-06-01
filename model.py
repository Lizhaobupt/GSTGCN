from cells import *
from math_utils import *
from os.path import join as pjoin
import time
from data_utils import *


class GSTGCN(object):

    def __init__(self, inputs, blocks, args, len_train, mean, std, true_loss=True):

        self.blocks = blocks
        self.args = args
        self.mean = mean
        self.std = std
        self.true_loss = true_loss
        self.inputs = inputs

        self.loss = 0
        self.output = 0
        self.opt_op = None

        self._build()
        self._loss()

        global_steps = tf.Variable(0, trainable=False)
        if len_train % self.args.batch_size == 0:
            epoch_step = len_train / self.args.batch_size
        else:
            epoch_step = int(len_train / self.args.batch_size) + 1
        # Learning rate decay with rate 0.7 every 5 epochs.
        lr = tf.train.exponential_decay(self.args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.8,
                                        staircase=True)
        self.optimizer = tf.train.RMSPropOptimizer(lr)
        step_op = tf.assign_add(global_steps, 1)
        with tf.control_dependencies([step_op]):
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        inputs = self.inputs[:, :self.args.n_pre+self.args.n_his, :, :]
        for i, block in enumerate(self.blocks):
            input_dim, output_dim = block
            inputs_time_conv = TimeConvolution(args=self.args, act=tf.nn.tanh, name=f'layer_t_{i}',
                                               input_dim=input_dim, output_dim=output_dim)(inputs)
            inputs_spatial_conv = GraphConvolution(args=self.args, act=tf.nn.relu, name=f'layer_s_{i}',
                                                   input_dim=input_dim, output_dim=output_dim)(inputs)

            inputs = tf.concat([inputs_time_conv, inputs_spatial_conv], axis=-1)

        trend_pre = inputs[:, :self.args.n_his, :, :]
        period_re = inputs[:, self.args.n_pre:, :, :]
        inputs_pre = (trend_pre + period_re) / 2
       #  inputs_pre = Causal_conv(args=self.args, act=tf.nn.relu, name='trans', input_dim=2*output_dim,
       #                          output_dim=2*output_dim, Kt=2)(inputs)
       #  print('>>>>>>>>>>>', inputs_pre.shape)
        # output layer
        output_time_conv = TimeConvolution(args=self.args, act=tf.nn.tanh, name='layer_out_t',
                input_dim=2*output_dim, output_dim=2*output_dim)(inputs_pre)
        output_mlp_1 = MLP(args=self.args, act=lambda x: x, name='layer_out_mlp_1',
                           input_dim=2*output_dim, output_dim=64)(output_time_conv)
        output_mlp_2 = MLP(args=self.args, act=lambda x: x, name='layer_out_mlp_2',
                           input_dim=64, output_dim=1, layer_norm=False)(output_mlp_1)
        tf.add_to_collection('y_pre', value=output_mlp_2)
       # print('>>>>>>>out shape', output_mlp_2.shape)
        self.output = output_mlp_2


    def _base_model(self):
        inputs = self.inputs[:, :self.args.n_pre+self.args.n_his, :, :]
        for i, block in enumerate(self.blocks):
            input_dim, output_dim = block

            inputs_time_conv = Causal_conv(args=self.args, act=tf.nn.tanh, name=f'layer_t_{i}',
                    input_dim=input_dim, output_dim=output_dim, Kt=[2, 3, 2])(inputs)
            inputs_spatial_conv = GraphConvolution(args=self.args, act=tf.nn.relu, name=f'layer_s_{i}',
                                                input_dim=input_dim, output_dim=output_dim)(inputs)
           
            inputs = tf.concat([inputs_time_conv, inputs_spatial_conv], axis=-1)
            
        
        # print('>>>>>>>>>>', inputs.shape)
        trend_pre = inputs[:, :self.args.n_his, :, :]
        period_re = inputs[:, self.args.n_pre:, :, :]
        inputs_pre = (trend_pre + period_re) / 2
        # output layer       
        output_time_conv = Causal_conv(args=self.args, act=tf.nn.tanh, name='layer_out_t',
                                                input_dim=2*output_dim, output_dim=2*output_dim, Kt=[2, 3, 2])(inputs_pre)
        print('>>>>>>>>>>>>', output_time_conv.shape)
        
        output_mlp_1 = MLP(args=self.args, act=lambda x: x, name='layer_out_mlp_1',
                input_dim=2*output_dim, output_dim=64)(output_time_conv)
                
        output_mlp_2 = MLP(args=self.args, act=lambda x: x, name='layer_out_mlp_2',
                input_dim=64, output_dim=1, layer_norm=False)(output_mlp_1)
        tf.add_to_collection('y_pre', value=output_mlp_2)
        print('>>>>>>>out shape', output_mlp_2.shape)
        
        self.output = output_mlp_2


    def _loss_base_model(self):
    
        labels = self.inputs[:, -self.args.n_pre:-self.args.n_pre+1, :, :]
        
        if self.true_loss:
            self.loss = tensor_mae(labels*self.std, self.output*self.std)
            
        else:
            self.loss = tensor_mae(labels, self.output)


    def _loss(self):
        labels = self.inputs[:, -self.args.n_pre:, :, :]
        print(labels.shape)
        if self.true_loss:
            self.loss = tensor_mae(labels*self.std, self.output*self.std)
        else:
            self.loss = tensor_mae(labels, self.output)

    def save_model(self, sess, model_name, save_path='./output'):
        saver = tf.train.Saver()
        prefix_path = saver.save(sess, pjoin(save_path, model_name))
        print(f'<< Saving model to {prefix_path} ...')

    def load_model(self, df_test, load_path='./output'):
        model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
        test_graph = tf.Graph()
        with test_graph.as_default():
            saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))
        with tf.Session(graph=test_graph) as test_sess:
            saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
            print(f'>> Loading saved model from {model_path} ...')
            pred = test_graph.get_collection('y_pre')
    
            # evl
            test_pre = self.evl_epoch(test_sess, df_test, pred)
        return test_pre

    def evl_epoch(self, sess, data_set, outputs):
        pred_list = []
        for i, df_batch in enumerate(data_batch(data_set, self.args.batch_size, shuffle=False)):
            pred = sess.run(outputs, feed_dict={'data_input:0': df_batch})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            pred_list.append(pred)

        pred_list = np.concatenate(pred_list, axis=0)
        return pred_list

    def evl_epoch_mul(self, sess, data_set, outputs):
    
        pred_list = []
    
        for i, df_batch in enumerate(data_batch(data_set, self.args.batch_size, shuffle=False)):    
            step_list = []
            for time_step in range(self.args.n_pre):
                pred = sess.run(outputs, feed_dict={'data_input:0': df_batch})
                if isinstance(pred, list):
                    pred = np.array(pred[0])
                df_batch[:, 0:self.args.n_pre+self.args.n_his-1, :, :] = df_batch[:, 1:self.args.n_his+self.args.n_pre, :, :]
                df_batch[:, self.args.n_pre+self.args.n_his-1:self.args.n_pre+self.args.n_his, :, :] = pred
                step_list.append(pred)
            step_list = np.concatenate(step_list, axis=1)
           # print(step_list.shape)
            pred_list.append(step_list)    

        pred_list = np.concatenate(pred_list, axis=0)
        print(pred_list.shape)
        return pred_list

    def train(self, df_train, df_val):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.args.epoch):
                time_start = time.time()
                for j, df_batch in enumerate(data_batch(df_train, self.args.batch_size, shuffle=True)):
                    train_op, train_loss = sess.run([self.opt_op, self.loss],
                                                    feed_dict={self.inputs: df_batch})
                    if j % 50 == 0:
                        print(f'epoch:{i}  train_loss:{train_loss:.4f}')

                print(f'training time {time.time() - time_start:.4f}s')

                val_pre = self.evl_epoch(sess, df_val, self.output)
                evl = evaluation(df_val[:, -self.args.n_pre:, :, :], val_pre, self.mean, self.std, self.args.n_pre)

                for time_id in range(self.args.n_pre):
                    print(f'mae_loss {evl[time_id,0]:.4f}, rmse_loss {evl[time_id, 1]:.4f},'
                          f' mape_loss {evl[time_id, 2]*100:.4f}%')

                print(f'mae_mean {np.mean(evl, axis=0)[0]:.4f}, '
                      f'rmse_mean {np.mean(evl, axis=0)[1]:.4f},'
                      f' mape_mean {np.mean(evl, axis=0)[2]*100:.4f}%')
            self.save_model(sess, 'GSTGCN')

    def test(self, df_test, load_path):

        test_pre = self.load_model(df_test, load_path)
        # print(test_pre.shape)
        evl = evaluation(df_test[:, -self.args.n_pre:, :, :], test_pre, self.mean, self.std, self.args.n_pre)

        for time_id in range(self.args.n_pre):
            print(f'mae_loss {evl[time_id, 0]:.4f}, rmse_loss {evl[time_id, 1]:.4f}, mape_loss {evl[time_id, 2] * 100:.4f}%')

        print(f'mae_mean {np.mean(evl, axis=0)[0]:.4f}, '
              f'rmse_mean {np.mean(evl, axis=0)[1]:.4f},'
              f' mape_mean {np.mean(evl, axis=0)[2] * 100:.4f}%')



