import os
from math_utils import *


def load_data(file_path):
    """
    load data
    :param file_path:
    :return: data[B, N]
    """
    # path
    data_path = os.path.join("data/", f'{file_path}', f'{file_path}.npz')
    data = np.load(data_path)
    print(f'load data from path: data/{file_path}/{file_path}.npz')
    # B, N
    data = data['data'][:, :, 0]
    return data


def data_spilt(data_set, data_ratio, offset, n_his, n_pre, n_route, day_slot, c_0=1):

    if offset == 0:
        # data size
        n_slot = int(len(data_set) * data_ratio - day_slot) - n_his - n_pre + 1
        # data scale
        n_scale = n_his + 2 * n_pre
        tmp_seq = np.zeros([int(n_slot), n_scale, n_route, c_0])
        for i in range(n_slot):
            # period
            sta1 = i + n_his
            end1 = int(sta1 + n_pre)
            df_period = np.reshape(data_set[sta1:end1, :], [n_pre, n_route, c_0])
            # trend
            sta2 = i + day_slot
            end2 = sta2 + n_his
            df_trend = np.reshape(data_set[sta2:end2, :], [n_his, n_route, c_0])
            # pre
            df_pre = np.reshape(data_set[end2:end2 + n_pre, :], [n_pre, n_route, c_0])
            df = np.concatenate([df_period, df_trend, df_pre], axis=0)
            tmp_seq[i, :, :, :] = df

    else:
        n_slot = int(len(data_set) * data_ratio) - n_his - n_pre + 1
        n_frame = n_his + 2 * n_pre
        tmp_seq = np.zeros([n_slot, n_frame, n_route, c_0])
        for i in range(n_slot):
            # period
            sta1 = i + n_his + int(offset * len(data_set)) - day_slot
            end1 = int(sta1 + n_pre)
            df_period = np.reshape(data_set[sta1:end1, :], [n_pre, n_route, c_0])
            # trend
            sta2 = i + int(offset * len(data_set))
            end2 = sta2 + n_his
            df_trend = np.reshape(data_set[sta2:end2, :], [n_his, n_route, c_0])
            df_pre = np.reshape(data_set[end2:end2 + n_pre, :], [n_pre, n_route, c_0])
            df = np.concatenate([df_period, df_trend, df_pre], axis=0)
            tmp_seq[i, :, :, :] = df

    return tmp_seq


def data_gen(file_path, data_assign, n_route, n_his, n_pre, day_slot):

    # data assign
    n_train, n_val, n_test = data_assign
    data_set = load_data(file_path)
    # train
    df_train = data_spilt(data_set, n_train, offset=0, n_his=n_his, n_pre=n_pre, n_route=n_route, day_slot=day_slot)
    # val
    df_val = data_spilt(data_set, n_val, offset=n_train, n_his=n_his, n_pre=n_pre, n_route=n_route, day_slot=day_slot)
    # test
    df_test = data_spilt(data_set, n_test, offset=n_train+n_val, n_his=n_his, n_pre=n_pre, n_route=n_route, day_slot=day_slot)

    df_mean = np.mean(df_train)
    df_std = np.std(df_train)
    # z_score
    df_train = z_score(df_train, df_mean, df_std)
    df_val = z_score(df_val, df_mean, df_std)
    df_test = z_score(df_test, df_mean, df_std)
    print('train dataset shape:',df_train.shape)
    print('val dataset shape:',df_val.shape)
    print('test dataset shape:',df_test.shape)
    print(f'mean:{df_mean}, std:{df_std}')

    return df_train, df_val, df_test, df_mean, df_std


def data_batch(data, batch_size, shuffle):
    """

    :param data:
    :param batch_size:
    :param shuffle:
    :return: shape [Batch_size, T, N, C_0]
    """
    data_len = len(data)
    data_id = np.arange(data_len)
    # shuffle
    if shuffle:
        np.random.shuffle(data_id)
        # data = data[data_id]

    for st_id in range(0, data_len, batch_size):
        end_id = st_id + batch_size
        if end_id > data_len:
            end_id = data_len

        yield data[data_id[st_id:end_id]]

