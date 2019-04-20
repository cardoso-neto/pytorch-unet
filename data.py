import argparse
from itertools import chain
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

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
        tiers: Sequence[str] = ('q40',),  # Order here should be highest to lowest quality, so the model trains on the easier pics first
        hq_extension: str = 'jpg',  # TODO generalize for several extensions
        lq_extension: str = 'jpg',
        flatten: bool = True,  # still not clear how on I'm going to use this...
        debug: bool = False,
    ):
        '''
        dataset_root_path: the root of the TieredJPEGs dataset.
        transforms_: list of transforms to compose and apply on each image.
        '''
        self.dataset_root = Path(dataset_root_path)
        self.transform = transforms.Compose(_transforms)
        self.split = split
        self.tiers = tiers
        self.debug = debug

        self.subfolders: List[Path] = sorted(self.dataset_root.glob(f'{self.split}*'))

        self.original_images: List[List[Path]] = [  # shape (num_folders, num_pictures_per_folder)
            sorted(
                self.dataset_root.joinpath(subfolder / 'hq').glob(f'*.{hq_extension}')
            )
            for subfolder in self.subfolders
        ]

        self.compressed_images: List[List[Path]] = [  # shape (num_subfolders * num_tiers, num_pictures_per_folder)
            sorted(
                self.dataset_root.joinpath(subfolder / quality).glob(
                    f'*.{lq_extension}'
                )
            )
            for subfolder in self.subfolders
            for quality in self.tiers
        ]

        if flatten:
            self.original_images = list(chain(*self.original_images))
            self.compressed_images = list(chain(*self.compressed_images))

    def __getitem__(
        self, index: int
    ) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        highq = self.original_images[index]
        # TODO deal with this. Return all of them? Create different loaders for each? idk
        lowq = self.compressed_images[index]
        print(lowq, highq)
        assert highq.stem == lowq.stem

        hq: torch.Tensor = self.transform(Image.open(highq).convert('RGB'))
        lq: torch.Tensor = self.transform(Image.open(lowq).convert('RGB'))

        return lq, hq

    def __len__(self):
        if self.debug:
            return 2
        return len(self.original_images)

    def __repr__(self):
        return f'TieredJPEGs<split={self.split}>'


transforms_dict = {
    'train': [
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
    'val': [
        # transforms.FiveCrop((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
    'test': [
        # transforms.FiveCrop((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
}


def get_loader(
    data_path: str,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    debug: bool = False,
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
        num_workers=num_workers,
    )

    return loader


if __name__ == '__main__':
    # dataset_root = '/home/nei/projs/hyper-resolution/jpeg-noise-remover/dataset/'
    dataset_root = '/home/nei/projs/hyper-resolution/pytorch-unet/data'
    # import params
    # args = params.get_data_params()

    loader = get_loader(dataset_root, 'train', batch_size=1)
    stuff = iter(loader).next()

    print(stuff, '\n')
    print(stuff[0].shape)
