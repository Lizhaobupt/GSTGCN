import tensorflow as tf


class Layer(object):

    def __init__(self, args, act, name, layer_norm=True):
        self.n_route = args.n_route
        self.n_his = args.n_his
        self.n_pre = args.n_pre
        self.encode_dim = args.encode_dim
        self.act = act
        self.name = name
        self.layer_norm = layer_norm

    def l_norm(self, x, name):

        _, _, N, C = x.get_shape().as_list()
        mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

        gamma = tf.get_variable(f'gamma_{name}', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable(f'beta_{name}', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
        return _x

    def _call(self, inputs):

        return inputs

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            outputs = self._call(inputs)

            return outputs


class MLP(Layer):

    def __init__(self, args, act, name, input_dim,  output_dim, layer_norm=True):
        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):

        _, T, _, _ = inputs.get_shape().as_list()
        # Parameter definition
        w_encode = tf.get_variable(name='w_encode', shape=[self.n_route, self.encode_dim], dtype=tf.float32)
        w_decode = tf.get_variable(name='w_decode',
                                   shape=[self.encode_dim, self.input_dim * self.output_dim], dtype=tf.float32)
        bias = tf.get_variable(name='bias', initializer=tf.zeros([self.output_dim]), dtype=tf.float32)
        weight = tf.reshape(tf.matmul(w_encode, w_decode), [self.n_route, self.input_dim, self.output_dim])
        # inputs[B, T, N, C] -> [B*T, N, C] -> [N, B*T, C]
        re_inputs = tf.transpose(tf.reshape(inputs, [-1, self.n_route, self.input_dim]), [1, 0 , 2])
        # inputs[N, B*T, C] -> [N, B*T, C_OUT] -> [B*T, N, C_OUT]
        mul_input = tf.transpose((tf.matmul(re_inputs, weight) + bias), [1, 0, 2])

        mul_input = tf.reshape(mul_input, [-1, T, self.n_route, self.output_dim])
        # layer_norm
        if self.layer_norm:
            mul_input = self.l_norm(mul_input, self.name)

        return self.act(mul_input)


class TimeConvolution(Layer):

    def __init__(self, args, act, name, input_dim, output_dim, layer_norm=True, if_trans=False):
        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.if_trans = if_trans
    
    def _conv_cell_1(self, res_inputs, input_dim, output_dim):

        wt1 = tf.get_variable(name='wt1_res', shape=[1, 1, input_dim, output_dim], dtype=tf.float32)
        bt1 = tf.get_variable(name='bt1_res', initializer=tf.zeros([output_dim]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(res_inputs, wt1, strides=[1, 1, 1, 1], padding='SAME') + bt1

        return x_conv

    def _conv_cell_2(self, inputs, res_inputs, input_dim, output_dim, layer_id):

        # first
        wt2_1 = tf.get_variable(name=f'wt1_{layer_id}', shape=[2, 1, input_dim, output_dim], dtype=tf.float32)
        bt2_1 = tf.get_variable(name=f'bt1_{layer_id}', initializer=tf.zeros([output_dim]), dtype=tf.float32)
        # two
        wt2_2 = tf.get_variable(name=f'wt2_{layer_id}', shape=[2, 1, input_dim, output_dim], dtype=tf.float32)
        bt2_2 = tf.get_variable(name=f'bt2_{layer_id}', initializer=tf.zeros([output_dim]), dtype=tf.float32)

        x_conv_1 = tf.nn.conv2d(inputs, wt2_1, strides=[1, 2, 1, 1], padding='VALID') + bt2_1
        x_conv_2 = tf.nn.conv2d(inputs, wt2_2, strides=[1, 2, 1, 1], padding='VALID') + bt2_2

        x_conv = tf.concat([x_conv_1, x_conv_2], axis=1)
        x_conv = x_conv + res_inputs

        if self.layer_norm:
            x_conv = self.l_norm(x_conv, layer_id)
        return x_conv

    def _conv_cell_3(self, inputs, res_inputs, input_dim, output_dim, layer_id):

        # first
        wt3_1 = tf.get_variable(name=f'wt1_{layer_id}', shape=[3, 1, input_dim, output_dim], dtype=tf.float32)
        bt3_1 = tf.get_variable(name=f'bt1_{layer_id}', initializer=tf.zeros([output_dim]), dtype=tf.float32)
        # two
        wt3_2 = tf.get_variable(name=f'wt2_{layer_id}', shape=[3, 1, input_dim, output_dim], dtype=tf.float32)
        bt3_2 = tf.get_variable(name=f'bt2_{layer_id}', initializer=tf.zeros([output_dim]), dtype=tf.float32)
        # three
        wt3_3 = tf.get_variable(name=f'wt3_{layer_id}', shape=[3, 1, input_dim, output_dim], dtype=tf.float32)
        bt3_3 = tf.get_variable(name=f'bt3_{layer_id}', initializer=tf.zeros([output_dim]), dtype=tf.float32)

        x_conv_1 = tf.nn.conv2d(inputs, wt3_1, strides=[1, 3, 1, 1], padding='VALID') + bt3_1
        x_conv_2 = tf.nn.conv2d(inputs, wt3_2, strides=[1, 3, 1, 1], padding='VALID') + bt3_2
        x_conv_3 = tf.nn.conv2d(inputs, wt3_3, strides=[1, 3, 1, 1], padding='VALID') + bt3_3
        # concat
        x_conv = tf.concat([x_conv_1, x_conv_2, x_conv_3], axis=1)
        x_conv = x_conv + res_inputs

        if self.layer_norm:
            x_conv = self.l_norm(x_conv, layer_id)

        return x_conv

    def _call(self, inputs):

        if self.input_dim == self.output_dim:
            res_inputs = inputs
        else:
            res_inputs = self._conv_cell_1(inputs, self.input_dim, self.output_dim)

        inputs_conv1 = self._conv_cell_2(inputs=inputs, res_inputs=res_inputs,
                                         input_dim=self.input_dim, output_dim=self.output_dim,
                                         layer_id=1)

        inputs_conv2 = self._conv_cell_3(inputs=inputs_conv1, res_inputs=res_inputs,
                                         input_dim=self.output_dim, output_dim=self.output_dim,
                                         layer_id=2)
        
        inputs_conv3 = self._conv_cell_2(inputs=inputs_conv2, res_inputs=res_inputs,
                                         input_dim=self.output_dim, output_dim=self.output_dim,
                                         layer_id=3)
        if self.if_trans:
            wt_trans = tf.get_variable(name='wt_trans', shape=[2, 1, self.output_dim, self.output_dim], dtype=tf.float32)
            inputs_conv3 = tf.nn.conv2d(inputs_conv3, wt_trans, strides=[1, 2, 1, 1], padding='SAME')
        return self.act(inputs_conv3)


class GraphConvolution(Layer):

    def __init__(self, args, act, name, input_dim, output_dim, layer_norm=True):
        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _gconv(self, inputs, inputs_res, theta, bias):
        _, T, N, C = inputs.get_shape().as_list()
        inputs_period, inputs_trend = tf.split(inputs, num_or_size_splits=2, axis=1)
        # B, T/2, 2*N, C
        inputs_concat = tf.concat([inputs_period, inputs_trend], axis=2)
        node_embedding = tf.get_variable(name='embedding', shape=[2*N, self.encode_dim], dtype=tf.float32)
        kernel = tf.matmul(node_embedding, tf.transpose(node_embedding, [1, 0]))
        kernel = tf.nn.softmax(tf.nn.relu(kernel), axis=0)
        # inputs[B, T, N, C] -> [B, T, C, N] -> [B*T*C, N]
        inputs_tmp = tf.reshape(tf.transpose(inputs_concat, [0, 1, 3, 2]), [-1, 2*N])
        # inputs[B*T*C, N] -> [B*T*C, N] -> [B*T, C, N]
        inputs_mul = tf.reshape(tf.matmul(inputs_tmp, kernel), [-1, C, 2*N])
        # inputs[B*T, C, N] -> [B*T, N, C] ->[B*T*N, C]
        inputs_ker = tf.reshape(tf.transpose(inputs_mul, [0, 2, 1]), [-1, C])
        # inputs -> [B, T, N, c_out]
        inputs_gconv = tf.reshape(tf.matmul(inputs_ker, theta), [-1, T//2, 2*N, self.output_dim]) 
        inputs_gconv_period, inputs_gconv_trend = tf.split(inputs_gconv, num_or_size_splits=2, axis=2)
        inputs_gconv = tf.concat([inputs_gconv_period, inputs_gconv_trend], axis=1) + bias
        inputs_gconv = inputs_gconv + inputs_res
        if self.layer_norm:
            inputs_gconv = self.l_norm(inputs_gconv, name=self.name)
        return inputs_gconv

    def _call(self, inputs):
        """
        Graph convolution
        :param inputs: [B, T, N, C_IN]
        :return: [B, T, N, C_OUT]
        """
        if self.input_dim == self.output_dim:
            inputs_res = inputs
        else:
            wt_res = tf.get_variable(name='wt_res', shape=[1, 1, self.input_dim, self.output_dim], dtype=tf.float32)
            inputs_res = tf.nn.conv2d(inputs, wt_res, strides=[1, 1, 1, 1], padding='SAME')

        ws_1 = tf.get_variable(name='ws', shape=[self.input_dim, self.output_dim], dtype=tf.float32)
        bs_1 = tf.get_variable(name='bs', initializer=tf.zeros([self.output_dim]), dtype=tf.float32)
        gconv_output = self._gconv(inputs=inputs, inputs_res=inputs_res, theta=ws_1, bias=bs_1)

        return self.act(gconv_output)


class Causal_conv(Layer):

    def __init__(self, args, act, name, input_dim, output_dim, Kt, layer_norm=True):
        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.Kt = Kt
 
    def _call(self, inputs):
        
        K1, K2, K3 = self.Kt
                
        # short cut
        
        if self.input_dim == self.output_dim:
            inputs_res = inputs
            
        else:
            wt_res = tf.get_variable(name='wt_res', shape=[1, 1, self.input_dim, self.output_dim], dtype=tf.float32)    
            inputs_res = tf.nn.conv2d(inputs, wt_res, strides=[1, 1, 1, 1], padding='SAME')
            
        # inputs_res = inputs_res[:, K1+K2+K3-3:, :, :]

        wt1 = tf.get_variable(name='wt1', shape=[K1, 1, self.input_dim, self.output_dim], dtype=tf.float32)
        bt1 = tf.get_variable(name='bt1', initializer=tf.zeros([self.output_dim]), dtype=tf.float32)

        wt2 = tf.get_variable(name='wt2', shape=[K2, 1, self.output_dim, self.output_dim], dtype=tf.float32)
        bt2 = tf.get_variable(name='bt2', initializer=tf.zeros([self.output_dim]), dtype=tf.float32)

        wt3 = tf.get_variable(name='wt3', shape=[K3, 1, self.output_dim, self.output_dim], dtype=tf.float32)
        bt3 = tf.get_variable(name='bt3', initializer=tf.zeros([self.output_dim]), dtype=tf.float32)

        inputs_conv1 = tf.nn.conv2d(inputs, wt1, strides=[1, 1, 1, 1], padding='SAME') + bt1
        inputs_conv1 = self.l_norm(inputs_conv1+inputs_res, name='l_nor_t_1')
        inputs_conv2 = tf.nn.conv2d(inputs_conv1, wt2, strides=[1, 1, 1, 1], padding='SAME') + bt2
        inputs_conv2 = self.l_norm(inputs_conv2+inputs_res, name='l_nor_t_2')
        inputs_conv3 = tf.nn.conv2d(inputs_conv2, wt3, strides=[1, 1, 1, 1], padding='SAME') + bt3
                        
        inputs_conv = self.l_norm(inputs_conv3+inputs_res, name='l_nor_t_3')

        return self.act(inputs_conv)


class Graph_base_conv(Layer):

    def __init__(self, args, act, name, input_dim, output_dim, layer_norm=True):
        
        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _gconv(self, inputs, inputs_res, theta, bias):
                            
        _, T, N, C = inputs.get_shape().as_list()
        node_embedding = tf.get_variable(name='embedding', shape=[N, self.encode_dim], dtype=tf.float32)
        kernel = tf.matmul(node_embedding, tf.transpose(node_embedding, [1, 0]))
        kernel = tf.nn.softmax(tf.nn.relu(kernel), axis=0)
        
        # inputs[B, T, N, C] -> [B, T, C, N] -> [B*T*C, N]
        inputs_tmp = tf.reshape(tf.transpose(inputs, [0, 1, 3, 2]), [-1, N])
        # inputs[B*T*C, N] -> [B*T*C, N] -> [B*T, C, N]
        inputs_mul = tf.reshape(tf.matmul(inputs_tmp, kernel), [-1, C, N])
        # inputs[B*T, C, N] -> [B*T, N, C] ->[B*T*N, C]
        inputs_ker = tf.reshape(tf.transpose(inputs_mul, [0, 2, 1]), [-1, C])
        # inputs -> [B, T, N, c_out]
        inputs_gconv = tf.reshape(tf.matmul(inputs_ker, theta), [-1, T, N, self.output_dim]) + bias
        inputs_gconv = inputs_gconv + inputs_res
        if self.layer_norm:
            inputs_gconv = self.l_norm(inputs_gconv, name=self.name)

        return inputs_gconv

            
    def _call(self, inputs):
    
        """
        Graph convolution
        :param inputs: [B, T, N, C_IN]
        :return: [B, T, N, C_OUT]
        """    
                
        if self.input_dim == self.output_dim:
            inputs_res = inputs
            
        else:
            wt_res = tf.get_variable(name='wt_res', shape=[1, 1, self.input_dim, self.output_dim], dtype=tf.float32)
            inputs_res = tf.nn.conv2d(inputs, wt_res, strides=[1, 1, 1, 1], padding='SAME')
        
        ws_1 = tf.get_variable(name='ws', shape=[self.input_dim, self.output_dim], dtype=tf.float32)    
        bs_1 = tf.get_variable(name='bs', initializer=tf.zeros([self.output_dim]), dtype=tf.float32)
    
        gconv_output = self._gconv(inputs=inputs, inputs_res=inputs_res, theta=ws_1, bias=bs_1)
    
        return self.act(gconv_output)



