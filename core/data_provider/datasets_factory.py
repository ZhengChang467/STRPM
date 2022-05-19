import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from core.data_provider.human36m import ToTensor, Norm


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
                  is_training=True,
                  is_shuffle=True):
    if dataset == 'human36m':
        from core.data_provider.human36m import human36m as data_set
    elif dataset == 'ucfsport':
        from core.data_provider.ucfsport import ucfsport as data_set
    elif dataset == 'sjtu4k':
        from core.data_provider.sjtu4k import sjtu4k as data_set

    if is_training:
        mode = 'train'
    else:
        mode = 'test'
    dataset = data_set(
        configs=configs,
        data_train_path=data_train_path,
        data_test_path=data_test_path,
        mode=mode,
        transform=transforms.Compose([Norm(), ToTensor()]))
    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=2)
