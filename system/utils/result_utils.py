import h5py
import numpy as np
import os


def average_data(file_paths):
    test_acc = []
    
    # 直接遍历精确路径并读取
    for file_path in file_paths:
        test_acc.append(np.array(read_data_then_delete(file_path, delete=False)))

    # 防止列表为空的情况
    if not test_acc:
        print("未找到任何实验结果数据。")
        return

    max_accuracy = []
    for acc in test_acc:
        if len(acc) > 0:
            max_accuracy.append(acc.max())

    if max_accuracy:
        print("std for best accuracy:", np.std(max_accuracy))
        print("mean for best accuracy:", np.mean(max_accuracy))


# def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
#     test_acc = []
#     algorithms_list = [algorithm] * times
#     for i in range(times):
#         file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
#         test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

#     return test_acc


def read_data_then_delete(file_path, delete=False):
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc