import numpy as np

from utils.tools import *
from nn.functional import sigmoid, img2col
# Attension:
# - Never change the value of input, which will change the result of backward


class operator(object):
    """
    operator abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operator):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operator):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class matmul(operator):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, input, weights):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(input, weights)

    def backward(self, out_grad, input, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward input with same shape as input
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        return in_grad, w_grad


class add_bias(operator):
    def __init__(self):
        super(add_bias, self).__init__()

    def forward(self, input, bias):
        '''
        # Arugments
          input: numpy array with shape (batch, in_features)
          bias: numpy array with shape (in_features)

        # Returns
          output: numpy array with shape(batch, in_features)
        '''
        return input + bias.reshape(1, -1)

    def backward(self, out_grad, input, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            bias: numpy array with shape (out_features)
        # Returns
            in_grad: gradient to the forward input with same shape as input
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = out_grad
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, b_grad


class linear(operator):
    def __init__(self):
        super(linear, self).__init__()
        self.matmul = matmul()
        self.add_bias = add_bias()

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        output = self.matmul.forward(input, weights)
        output = self.add_bias.forward(output, bias)
        # output = np.matmul(input, weights) + bias.reshape(1, -1)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            in_grad: gradient to the forward input of linear layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        # in_grad = np.matmul(out_grad, weights.T)
        # w_grad = np.matmul(input.T, out_grad)
        # b_grad = np.sum(out_grad, axis=0)
        out_grad, b_grad = self.add_bias.backward(out_grad, input, bias)
        in_grad, w_grad = self.matmul.backward(out_grad, input, weights)
        return in_grad, w_grad, b_grad


class conv(operator):
    def __init__(self, conv_params):
        """
        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The total number of 0s to be added along the height (or width) dimension; half of the 0s are added on the top (or left) and half at the bottom (or right). we will only test even numbers.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
        """
        super(conv, self).__init__()
        self.conv_params = conv_params

    def get_output_size(n, k, p, s):
        """
        # Arguments
            n: input size
            p: padding
            k: kernel size
            s: stride

        # Returns 
            o: output size
        """
        return int(np.floor((n + p - k) / s) + 1)

    def get_im2col_data_indexes(N, c_i, n_h, n_w, k_h, k_w, p, s):
        """
        # Arguments
            N: batch size (number of instances)
            c_i: number of input channels
            n_h: input height
            n_w: input width
            k_h: kernel height
            k_w: kernel width
            p: total padding
            s: stride length

        # Returns
            channel_idxs: indices of the channel in the img2col array row
            height_idxs: indices of the input height in the img2col array cell
            width_ixds: indices of the input width in the img2col array cell
        """
        o_h = self.get_output_size(n_h, k_h, p, s)
        o_w = self.get_output_size(n_w, k_w, p, s)

        # Indices of the channel for each row in the output reshaped batch data
        channel_idxs = np.repeat(np.arange(c_i), k_h * k_w).reshape(-1, 1)

        # Indices of heights of a single receptive field for all channels
        k_height_idxs = np.repeat(np.arange(k_h), k_w)
        k_height_idxs = np.tile(k_height_idxs, c_i)
        # Offsets for heights of receptive fields
        height_offsets = np.repeat(np.arange(o_h), o_w) * s

        height_idxs = k_height_idxs.reshape(-1, 1) + height_offsets.reshape(1, -1)


        # Indices of widths of a single receptive field for all channels
        k_width_idxs = np.tile(np.arange(k_w), k_h * c_i)
        width_offsets = np.tile(np.arange(o_w), o_h) * s

        width_ixds = k_width_idxs.reshape(-1, 1) + width_offsets.reshape(1, -1)
        
        return channel_idxs, height_idxs, width_ixds


    def img2col(X, N, c_i, n_h, n_w, k_h, k_w, p, s):
        """
        # Arguments
            X: padded input array
            N: batch size (number of instances)
            c_i: number of input channels
            n_h: input height
            n_w: input width
            k_h: kernel height
            k_w: kernel width
            p: total padding
            s: stride length

        # Returns
            X_hat: img2col representation of input of conv layer
            o_h: height of output
            o_w: width of output
        """
        channel_idxs, height_idxs, width_ixds = self.get_im2col_data_indexes(N, c_i, n_h, n_w, k_h, k_w, p, s)

        X_hat_batch = X[:, channel_idxs, height_idxs, width_ixds]
        # Combine the batch instances into a single matrix
        X_hat = X_hat_batch.transpose(1, 2, 0).reshape(k_h * k_w * c_i, -1)

        return X_hat, o_h, o_w


    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            output: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel'] # equivalent to number of kernels

        batch, in_channel, in_height, in_width = input.shape
        #####################################################################################
        # code here

        # Add zero padding to height and width dimensions of input
        p = int(pad / 2) # pad is guaranteed to be even
        X_pad = np.pad(input, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)

        # Reshape input feature maps and filters    
        X_hat, out_height, out_width = self.img2col(X_pad, batch, in_channel, in_height, in_width, kernel_h, kernel_w, pad, stride)
        W = weights.reshape((out_channel, in_channel * kernel_h * kernel_w))
        b = np.repeat(bias, batch * out_height * out_width).reshape(out_channel, -1)

        # Compute output
        Y = W.dot(X_hat) + b

        # Reshape output to correct shape
        output = Y.reshape(out_channel, out_height, out_width, batch)
        output = output.transpose(3, 0, 1, 2)
        #####################################################################################
        return output


    def col2img(dX_hat, N, c_i, n_h, n_w, k_h, k_w, p, s):
        """
        # Arguments
            dX_hat: img2col representation of gradient to the forward input of conv layer
            N: batch size (number of instances)
            c_i: number of input channels
            n_h: input height
            n_w: input width
            k_h: kernel height
            k_w: kernel width
            p: total padding
            s: stride length

        # Returns
            dX: gradient to the forward input of conv layer, with shape (N, c_i, n_h, n_w)
        """
        # Empty array to fill with gradient values
        dX_pad = np.zeros((N, c_i, n_h + p, n_w + p)) # TODO: , dtype=dX_hat.dtype)
        
        channel_idxs, height_idxs, width_ixds = self.get_im2col_data_indexes(N, c_i, n_h, n_w, k_h, k_w, p, s)

        # Get img2col for each batch instance
        dX_hat_batch = dX_hat.reshape(c_i * k_h * k_w, -1, N)
        dX_hat_batch = dX_hat_batch.transpose(2, 0, 1)

        # Fill in output with values from img2col batches
        np.add.at(dX_pad, (slice(None), channel_idxs, height_idxs, width_ixds), dX_hat_batch)

        # Remove padding rows and columns
        if p != 0:
            dX = dX_pad[:, :, p:-p, p:-p]
        else:
            dX = dX_pad
        
        return dX

    
    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            in_grad: gradient to the forward input of conv layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']
        
        
        
        batch, in_channel, in_height, in_width = input.shape
        #################################################################################
        # code here
        # Reshape input, weights, and gradient to col
        p = int(pad / 2) # pad is guaranteed to be even
        X_pad = np.pad(input, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
        X_hat, out_height, out_width = self.img2col(X_pad, batch, in_channel, in_height, in_width, kernel_h, kernel_w, pad, stride)

        W = weights.reshape((out_channel, in_channel * kernel_h * kernel_w))
        
        out_grad_col = out_grad.transpose(1, 2, 3, 0).reshape((out_channel, batch * out_height * out_width))

        # Compute gradients
        dX_hat = W.transpose().dot(out_grad_col)
        in_grad = self.col2img(dX_hat, batch, in_channel, in_height, in_width, kernel_h, kernel_w, pad, stride)
        
        w_grad = out_grad_col.dot(X_hat.transpose())
        w_grad = w_grad.reshape((out_channel, in_channel, kernel_h, kernel_w))

        b_grad = np.sum(out_grad, axis=(0, 2, 3)) # sum up for all batches, heights and widths in a channel
        #################################################################################
        return in_grad, w_grad, b_grad


class pool(operator):
    def __init__(self, pool_params):
        """
        # Arguments
            pool_params: dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The total number of 0s to be added along the height (or width) dimension; half of the 0s are added on the top (or left) and half at the bottom (or right). we will only test even numbers.
        """
        super(pool, self).__init__()
        self.pool_params = pool_params

    def forward(self, input):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        batch, in_channel, in_height, in_width = input.shape
        #####################################################################################
        # code here
        output = None
        #####################################################################################
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            in_grad: gradient to the forward input of pool layer, with same shape as input
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - pool_height + pad) // stride
        out_width = 1 + (in_width - pool_width + pad) // stride

        pad_scheme = (pad//2, pad - pad//2)
        input_pad = np.pad(input, pad_width=((0,0), (0,0), pad_scheme, pad_scheme),
                           mode='constant', constant_values=0)

        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        input_pool = img2col(input_pad, recep_fields_h,
                             recep_fields_w, pool_height, pool_width)
        input_pool = input_pool.reshape(
            batch, in_channel, -1, out_height, out_width)

        if pool_type == 'max':
            input_pool_grad = (input_pool == np.max(input_pool, axis=2, keepdims=True)) * \
                out_grad[:, :, np.newaxis, :, :]

        elif pool_type == 'avg':
            scale = 1 / (pool_height*pool_width)
            input_pool_grad = scale * \
                np.repeat(out_grad[:, :, np.newaxis, :, :],
                          pool_height*pool_width, axis=2)

        input_pool_grad = input_pool_grad.reshape(
            batch, in_channel, -1, out_height*out_width)

        input_pad_grad = np.zeros(input_pad.shape)
        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                input_pad_grad[:, :, i:i+pool_height, j:j+pool_width] += \
                    input_pool_grad[:, :, :, idx].reshape(
                        batch, in_channel, pool_height, pool_width)
                idx += 1
        in_grad = input_pad_grad[:, :, pad:pad+in_height, pad:pad+in_width]
        return in_grad


class dropout(operator):
    def __init__(self, rate, training=True, seed=None):
        """
        # Arguments
            rate: float[0, 1], the probability of setting a neuron to zero
            training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
        """
        self.rate = rate
        self.seed = seed
        self.training = training
        self.mask = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with any shape

        # Returns
            output: same shape as input
        """
        if self.training:
            scale = 1/(1-self.rate)
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            # Please use p as the probability to decide whether drop or not
            self.mask = (p >= self.rate).astype('int')
            #####################################################################################
            # code here
            output = None
            #####################################################################################
        else:
            output = input
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to forward output of dropout, same shape as input
            input: numpy array with any shape
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

        # Returns
            in_grad: gradient to forward input of dropout, same shape as input
        """
        if self.training:
            #####################################################################################
            # code here
            in_grad = None
            #####################################################################################
        else:
            in_grad = out_grad
        return in_grad


class vanilla_rnn(operator):
    def __init__(self):
        """
        # Arguments
            in_features: int, the number of inputs features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(vanilla_rnn, self).__init__()

    def forward(self, input, kernel, recurrent_kernel, bias):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        x, prev_h = input
        output = np.tanh(x.dot(kernel) + prev_h.dot(recurrent_kernel) + bias)
        return output

    def backward(self, out_grad, input, kernel, recurrent_kernel, bias):
        """
        # Arguments
            in_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            out_grads: [gradients to input numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
        """
        x, prev_h = input
        tanh_grad = np.nan_to_num(
            out_grad*(1-np.square(self.forward(input, kernel, recurrent_kernel, bias))))

        in_grad = [np.matmul(tanh_grad, kernel.T), np.matmul(
            tanh_grad, recurrent_kernel.T)]
        kernel_grad = np.matmul(np.nan_to_num(x.T), tanh_grad)
        r_kernel_grad = np.matmul(np.nan_to_num(prev_h.T), tanh_grad)
        b_grad = np.sum(tanh_grad, axis=0)

        return in_grad, kernel_grad, r_kernel_grad, b_grad


class gru(operator):
    def __init__(self):
        """
        # Arguments
            in_features: int, the number of inputs features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(gru, self).__init__()

    def forward(self, input, kernel, recurrent_kernel):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        x, prev_h = input
        _, all_units = kernel.shape
        units = all_units // 3
        kernel_z, kernel_r, kernel_h = kernel[:, :units], kernel[:, units:2*units],  kernel[:, 2*units:all_units]
        recurrent_kernel_z = recurrent_kernel[:, :units]
        recurrent_kernel_r = recurrent_kernel[:, units:2*units]
        recurrent_kernel_h = recurrent_kernel[:, 2*units:all_units]

        #####################################################################################
        # code here
        # reset gate
        x_r = None
        # update gate
        x_z = None
        # new gate
        x_h = None
        #####################################################################################

        output = (1 - x_z) * x_h + x_z * prev_h
        
        return output

    def backward(self, out_grad, input, kernel, recurrent_kernel):
        """
        # Arguments
            in_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            out_grads: [gradients to input numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
        """
        x, prev_h = input
        _, all_units = kernel.shape
        units = all_units // 3
        kernel_z, kernel_r, kernel_h = kernel[:, :units], kernel[:, units:2*units],  kernel[:, 2*units:all_units]
        recurrent_kernel_z = recurrent_kernel[:, :units]
        recurrent_kernel_r = recurrent_kernel[:, units:2*units]
        recurrent_kernel_h = recurrent_kernel[:, 2*units:all_units]

        #####################################################################################
        # code here
        x_grad = None
        prev_h_grad = None

        kernel_r_grad = None
        kernel_z_grad = None
        kernel_h_grad = None

        recurrent_kernel_r_grad = None
        recurrent_kernel_z_grad = None
        recurrent_kernel_h_grad = None
        #####################################################################################

        in_grad = [x_grad, prev_h_grad]
        kernel_grad = np.concatenate([kernel_z_grad, kernel_r_grad, kernel_h_grad], axis=-1)
        r_kernel_grad = np.concatenate([recurrent_kernel_z_grad, recurrent_kernel_r_grad, recurrent_kernel_h_grad], axis=-1)

        return in_grad, kernel_grad, r_kernel_grad


class softmax_cross_entropy(operator):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad

        