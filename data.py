import argparse
import glob
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Mapping, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from superjson import json
from torch.utils.data import DataLoader, Dataset


def read_file(filepath: Union[Path, str]) -> List[str]:
    with open(filepath) as file_obj:
        return file_obj.readlines()


def read_json(filepath: Union[Path, str]) -> Union[dict, list]:
    return json.load(str(filepath))


def save_json(
    json_obj: Union[Mapping, Sequence],
    filepath: Union[Path, str],
) -> None:
    json.dump(json_obj, str(filepath), overwrite=True)


class TieredJPEGs(Dataset):
    '''Images and their JPEG-compressed versions with different levels of quality.'''
    def __init__(
        self,
        dataset_root_path: str,
        _transforms: list = None,
        split: str = 'train',
        debug: bool = False,
    ):
        '''
        dataset_root_path: the root of the TieredJPEGs dataset.
        transforms_: list of transforms to compose and apply on each image.
        '''
        self.dataset_root = Path(dataset_root_path)
        self.transform = transforms.Compose(_transforms)
        self.split = split
        self.debug = debug

        self.ground_truth_paths: List[str] = sorted((self.dataset_root / 'hq').glob('*.png'))
        self.q20_paths = sorted((self.dataset_root / 'q20').glob('*.jpg'))

    def __getitem__(self, index: int) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[str]]:
        g_truth = self.ground_truth_paths[index]
        lq = self.q20_paths[index]
        assert g_truth.stem == lq.stem

        good: torch.Tensor = self.transform(Image.open(g_truth).convert('RGB'))
        bad: torch.Tensor = self.transform(Image.open(lq).convert('RGB'))

        return bad, good, lq.stem

    def __len__(self):
        if self.debug:
            return 2
        return len(self.ground_truth_paths)

    def __repr__(self):
        return f'TieredJPEGs<split={self.split}>'


transforms_dict = {
    'train': [
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
    'val': [
        transforms.FiveCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
    'test': [
        transforms.FiveCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
}


def get_loader(
    data_path: str,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    debug=False,
) -> DataLoader:

    ds = TieredJPEGs(
        dataset_root_path=data_path,
        _transforms=transforms_dict[split],
        split=split,
        debug=debug,
    )

    loader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers
    )

    return loader


if __name__ == '__main__':
    dataset_root = '/home/nei/projs/hyper-resolution/jpeg-noise-remover/dataset/'
    # import params
    # args = params.get_data_params()

    loader = get_loader(dataset_root, 'train', batch_size=1)
    stuff = iter(loader).next()

    print(stuff, '\n')
    print(stuff[0].shape)
    print(stuff[2])
