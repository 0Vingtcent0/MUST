import os
import random

import numpy as np
import torch
from monai.data import Dataset
from torch.utils.data import DataLoader

from .transforms import test_transforms, train_transforms, valid_transforms, unsup_base_transforms
from .utils import path_decoder, split_train_valid

__all__ = [
    "get_dataloaders_labeled",
    "get_dataloaders_unlabeled",
    "make_generator",
    "seed_worker",
]


def seed_worker(worker_id):
    """Seed each DataLoader worker from the DataLoader generator seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    try:
        from monai.utils import set_determinism

        set_determinism(seed=worker_seed)
    except Exception:
        pass
    try:
        from .transforms import set_transforms_random_state

        set_transforms_random_state(worker_seed)
    except Exception:
        pass


def make_generator(seed):
    """Create a deterministic torch generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator




def _record_key(record):
    """Return a stable key for detecting overlap between train and validation records."""
    return os.path.abspath(record["img"])


def _remove_validation_overlap(train_records, valid_records):
    """Remove explicit validation images from the training records.

    This keeps the training and validation splits disjoint when both are derived
    from the same full training mapping. It is especially useful for the 90% supervised protocol, where the full
    official training mapping can be used as
    the labeled source while a fixed 10% validation mapping is held out.
    """
    if not valid_records:
        return train_records, 0

    valid_keys = {_record_key(record) for record in valid_records}
    filtered_train = [record for record in train_records if _record_key(record) not in valid_keys]
    removed_count = len(train_records) - len(filtered_train)
    return filtered_train, removed_count


def _load_validation_records(root, mapping_file_valid, train_records, valid_portion, modality=None, seed=0):
    """Load an explicit validation mapping or split it from training records."""
    if mapping_file_valid:
        valid_records = path_decoder(root, mapping_file_valid, no_label=False, modality=modality)
        train_records, removed_count = _remove_validation_overlap(train_records, valid_records)
        if removed_count > 0:
            print(
                f"\n(DataLoaded) Removed {removed_count} validation image(s) "
                "from the training set to keep train/valid disjoint."
            )
        print(
            "\n(DataLoaded) Training data size: %d, Validation data size: %d\n"
            % (len(train_records), len(valid_records))
        )
        return train_records, valid_records

    return split_train_valid(train_records, valid_portion=valid_portion, seed=seed)


def get_dataloaders_labeled(
    root,
    mapping_file,
    mapping_file_test=None,
    mapping_file_valid=None,
    valid_portion=0.0,
    batch_size=8,
    modality=None,
    seed=0,
):
    """Create dataloaders for labeled training, validation, and test data."""
    data_dicts = path_decoder(root, mapping_file, modality=modality)
    random.Random(int(seed)).shuffle(data_dicts)

    train_dicts, valid_dicts = _load_validation_records(
        root=root,
        mapping_file_valid=mapping_file_valid,
        train_records=data_dicts,
        valid_portion=valid_portion,
        modality=modality,
        seed=seed,
    )

    test_dicts = []
    if mapping_file_test:
        test_dicts = path_decoder(root, mapping_file_test, no_label=False, modality=modality)
        print(f"\n(DataLoaded) Test data size: {len(test_dicts)}\n")

    trainset = Dataset(train_dicts, transform=train_transforms)
    validset = Dataset(valid_dicts, transform=valid_transforms)
    testset = Dataset(test_dicts, transform=test_transforms)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 1000),
    )
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 2000),
    )

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


def get_dataloaders_unlabeled(
    root,
    mapping_file,
    batch_size=8,
    shuffle=False,
    num_workers=5,
    modality=None,
    seed=0,
    return_images=False,
    transform=None,
):
    """Create a dataloader for unlabeled data.

    By default, the loader returns image paths for pseudo-label scanning.
    When ``return_images`` is true, it returns transformed images for online
    consistency baselines such as Mean Teacher and FixMatch-style training.
    """
    items = path_decoder(root, mapping_file, no_label=True, unlabeled=True, modality=modality)
    print(f"\n(DataLoaded) Unlabeled data size: {len(items)}\n")

    records = []
    for item in items:
        # path_decoder() has already resolved mapping paths against `root`.
        # Do not join `root` again here, otherwise paths become e.g.
        # ./data/CellSeg/data/CellSeg/Official/... in the online SSL loader.
        image_path = item["img"]
        if return_images:
            records.append({"img": image_path, "img_path": image_path})
        else:
            records.append({"img_path": image_path, "raw_img_path": image_path})

    if return_images and transform is None:
        transform = unsup_base_transforms()

    dataset = Dataset(records, transform=transform if return_images else None)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    return dataloader
