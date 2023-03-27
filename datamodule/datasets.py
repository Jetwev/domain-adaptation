import torch.utils.data as torchdata
import torchvision.transforms as transforms
from torchvision import datasets


class ConcatDataset(torchdata.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_datasets(folder, params):
    if folder in ['amazon', 'dslr', 'webcam']:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
        }

        train = datasets.ImageFolder(
            params.data_path + '/office31/' + folder + '/', data_transforms['train'])
        test = datasets.ImageFolder(
            params.data_path + '/office31/' + folder + '/', data_transforms['test'])
        print(f'{folder} train set size: {len(train)}')
        print(f'{folder} test set size: {len(test)}')
    else:
        raise ValueError(f'Dataset {folder} not found!')
    return train, test


def get_dataloaders(dataset_name, params):
    loader = torchdata.DataLoader(dataset_name, batch_size=params.batch_size,
                                  shuffle=params.shuffle, drop_last=True, num_workers=params.nb_wokers)
    return loader
