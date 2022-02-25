import contextlib


def tensor2numpy(tensor):
    try:
        tensor = tensor.cpu().numpy()
    except:
        pass
    return tensor
