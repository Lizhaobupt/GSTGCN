import numpy as np
import tensorflow as tf


def z_score(data, mean, std):
    return (data - mean) / std


def inverse_score(data, mean, std):
    return data * std + mean


def tensor_mae(true, pre):
    return tf.reduce_mean(tf.abs(true - pre))


def tensor_rmse(true, pre):
    return tf.sqrt(tf.reduce_mean((true - pre) ** 2))


def np_mape(true, pre, mask=True):
    if mask:
        mask_idx = np.where(true > 0)
        true = true[mask_idx]
        pre = pre[mask_idx]
    return np.mean(np.abs(np.divide((true - pre), true)))


def np_mae(true, pre):
    return np.mean(np.abs(true - pre))


def np_rmse(true, pre):
    return np.sqrt(np.mean((true - pre) ** 2))


def evaluation(true, pre, mean, std, n_pre):
    true_inverse = inverse_score(true, mean, std)
    pre_inverse = inverse_score(pre, mean, std)

    true_inverse = np.squeeze(true_inverse)
    pre_inverse = np.squeeze(pre_inverse)
   # print('>>>>>>>>>>>>>',true_inverse.shape)
   #  print('>>>>>>>>>>>>>', pre_inverse.shape)
    metrics = []
    for i in range(n_pre):
        x_true = true_inverse[:, i, :]
        x_pre = pre_inverse[:, i, :]
        x_mae = np_mae(x_true, x_pre)
        x_rmse = np_rmse(x_true, x_pre)
        x_mape = np_mape(x_true, x_pre)
        metrics.append([x_mae, x_rmse, x_mape])

    return np.array(metrics)
