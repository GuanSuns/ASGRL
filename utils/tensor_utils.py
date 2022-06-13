from pynvml import *
import torch
from torch.autograd import Variable
import math

# the default GPU
GPU_DEVICE_ID = torch.cuda.device_count() - 1


def auto_select_gpu():
    """
    Automatically select the gpu device with the max available memory space
    :return: the device id
    """
    if torch.cuda.is_available():
        global GPU_DEVICE_ID

        n_device = torch.cuda.device_count()
        max_memory = -math.inf
        nvmlInit()
        for i in range(min(n_device, 8)):
            device_id = n_device - 1 - i
            h = nvmlDeviceGetHandleByIndex(device_id)
            info = nvmlDeviceGetMemoryInfo(h)
            free_memory_space = int(info.free)
            print('[INFO] GPU device ', device_id, '- memory free: ', free_memory_space)
            if free_memory_space > max_memory:
                max_memory = free_memory_space
                GPU_DEVICE_ID = device_id
        print('\n[INFO] Auto select GPU device ', str(GPU_DEVICE_ID), '- memory free: ', max_memory)


auto_select_gpu()


def get_device_id():
    return GPU_DEVICE_ID


def get_device():
    return torch.device("cuda:"+str(GPU_DEVICE_ID) if torch.cuda.is_available() else "cpu")


def log_sum_exp(value, dim=None, keep_dim=False):
    """
    Numerically stable implementation of the operation
    value.exp().sum(dim, keep_dim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keep_dim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keep_dim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, (int, float, complex)):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def running_average_tensor_list(first_list, second_list, rate):
    """
    Return the result of
    first_list * (1 - rate) + second_list * rate
    Parameter
    ---------
    first_list (list) : A list of pytorch Tensors
    second_list (list) : A list of pytorch Tensors, should have the same
        format as first_list.
    rate (float): a learning rate, in [0, 1]
    Returns
    -------
    results (list): A list of Tensors with computed results.
    """
    results = []
    assert len(first_list) == len(second_list)
    for first_t, second_t in zip(first_list, second_list):
        assert first_t.shape == second_t.shape
        result_tensor = first_t * (1 - rate) + second_t * rate
        results.append(result_tensor)
    return results


def constant(value):
    """
    Return a torch Variable for computation. This is a function to help
    write short code.
    pytorch require multiplication take either two variables
    or two tensors. And it is recommended to wrap a constant
    in a variable which is kind of silly to me.
    https://discuss.pytorch.org/t/adding-a-scalar/218
    Parameters
    ----------
    value (float): The value to be wrapped in Variable
    Returns
    -------
    constant (Variable): The Variable wrapped the value for computation.
    """
    # noinspection PyArgumentList
    return Variable(torch.Tensor([value])).type(torch.float)


def freeze_parameters(module_list):
    for module in module_list:
        for param in module.parameters():
            param.requires_grad = False


def unfreeze_parameters(module_list):
    for module in module_list:
        for param in module.parameters():
            param.requires_grad = True


def get_parameters(module_list):
    parameters = []
    for module in module_list:
        parameters.extend(list(module.parameters()))
    return parameters


def binarize_tensor(tensor, threshold):
    tensor[tensor >= threshold] = 1.0
    tensor[tensor < threshold] = 0.0
    return tensor


def tensors_to_numpy(tensors):
    np_tensors = []
    for tensor in tensors:
        np_tensors.append(tensor.detach().cpu().numpy())
    return np_tensors


def identity(x, dim=0):
    """
    Return input without any change.
    x: torch.Tensor
    :return: torch.Tensor
    """
    return x