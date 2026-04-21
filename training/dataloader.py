import os
from pathlib import Path
import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from training.data_utils import (
    CTPETTransform,
    read_files_dicom,
)

class CTPETDataset(Dataset):
    """Dataset for paired CT/PET DICOM slices."""

    def __init__(self, args, fold):
        super().__init__()
        self.fold = fold
        self.dataset_name = args.dataset_name
        self.image_path = Path(args.image_path)
        self.ct_dir = self.image_path / "Combined_CT" / "Combined_CT"
        self.pet_dir = self.image_path / "Combined_PET" / "Combined_PET"
        assert self.ct_dir.exists(), f"Missing CT directory: {self.ct_dir}"
        assert self.pet_dir.exists(), f"Missing PET directory: {self.pet_dir}"

        ct_files = {path.name for path in self.ct_dir.glob("*.dcm")}
        pet_files = {path.name for path in self.pet_dir.glob("*.dcm")}
        paired_files = np.array(sorted(ct_files & pet_files))
        assert len(paired_files) > 0, "No paired CT/PET DICOM files found."

        rng = np.random.default_rng(args.seed)
        paired_files = paired_files[rng.permutation(len(paired_files))]
        test_split = float(getattr(args, "test_split", 0.1))
        split_idx = max(1, int(len(paired_files) * (1.0 - test_split)))
        if split_idx >= len(paired_files):
            split_idx = len(paired_files) - 1

        self.file_names = paired_files[:split_idx] if fold == "train" else paired_files[split_idx:]
        self.transform = CTPETTransform(
            image_size=getattr(args, "image_size", 256),
            augment=args.augment_train and fold == "train",
            normalize=args.normalize,
        )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        item = read_files_dicom(
            file_name=file_name,
            ct_dir=self.ct_dir,
            pet_dir=self.pet_dir,
            transform=self.transform,
            dataset_name=self.dataset_name,
        )
        item["idx_ct"] = idx
        item["idx_pet"] = idx
        return item

class CTPETDataLoader(LightningDataModule):
    """
    General data loader class for PyTorch Lightning.

    This class handles the creation of data loaders for training and testing, 
    including the initialization of datasets and batch processing.
    """
    
    def __init__(self, args):

        """
        Initialize the CellDataLoader instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataloader configuration.
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.init_dataset()

    def init_dataset(self):
        """
        Initialize dataset and data loaders.
        """
        self.training_set, self.test_set = self.create_torch_datasets()
        sampler_train = torch.utils.data.DistributedSampler(
            self.training_set, num_replicas=self.args.num_tasks, rank=self.args.global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            self.test_set, num_replicas=self.args.num_tasks, rank=self.args.global_rank, shuffle=False
        )
        self.loader_train = torch.utils.data.DataLoader(self.training_set, 
                                                        sampler=sampler_train,
                                                        batch_size=self.args.batch_size, 
                                                        num_workers=self.args.num_workers, 
                                                        pin_memory=self.args.pin_mem,
                                                        drop_last=True)  
        self.loader_test = torch.utils.data.DataLoader(self.test_set, 
                                                       sampler=sampler_test,
                                                       batch_size=self.args.batch_size, 
                                                       num_workers=self.args.num_workers, 
                                                       drop_last=False)          

    def create_torch_datasets(self):
        """
        Create datasets compatible with the PyTorch training loop.
        
        Returns:
            tuple: Training and test datasets.
        """
        assert self.args.dataset == "ctpet" or self.args.dataset_name == "ctpet", "Only 'ctpet' dataset is supported in CTPETDataLoader."
        training_set = CTPETDataset(self.args, fold="train")
        test_set = CTPETDataset(self.args, fold="test") 
        return training_set, test_set
    
    def train_dataloader(self):
        """
        Return the training data loader.
        
        Returns:
            DataLoader: Training data loader.
        """
        return self.loader_train
    
    def val_dataloader(self):
        """
        Return the validation data loader.
        
        Returns:
            DataLoader: Validation data loader.
        """
        return self.loader_test
    
    def test_dataloader(self):
        """
        Return the test data loader.
        
        Returns:
            DataLoader: Test data loader.
        """
        return self.loader_test

class CTPETDataLoader_Eval(LightningDataModule):
    """
    General data loader class for PyTorch Lightning.

    This class handles the creation of data loaders for training and testing, 
    including the initialization of datasets and batch processing.
    """
    
    def __init__(self, args):

        """
        Initialize the CellDataLoader instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataloader configuration.
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.init_dataset()

    def init_dataset(self):
        """
        Initialize dataset and data loaders.
        """
        self.training_set, self.test_set = self.create_torch_datasets()
        self.loader_train = torch.utils.data.DataLoader(self.training_set, 
                                                        shuffle=True,
                                                        batch_size=self.args.batch_size, 
                                                        num_workers=self.args.num_workers, 
                                                        pin_memory=self.args.pin_mem,
                                                        drop_last=True)  
        self.loader_test = torch.utils.data.DataLoader(self.test_set, 
                                                       shuffle=False,
                                                       batch_size=self.args.batch_size, 
                                                       num_workers=self.args.num_workers, 
                                                       drop_last=False)          

    def create_torch_datasets(self):
        """
        Create datasets compatible with the PyTorch training loop.
        
        Returns:
            tuple: Training and test datasets.
        """
        assert self.args.dataset == "ctpet" or self.args.dataset_name == "ctpet", "Only 'ctpet' dataset is supported in CTPETDataLoader."
        training_set = CTPETDataset(self.args, fold="train")
        test_set = CTPETDataset(self.args, fold="test")
        return training_set, test_set

    def train_dataloader(self):
        """
        Return the training data loader.
        
        Returns:
            DataLoader: Training data loader.
        """
        return self.loader_train
    
    def val_dataloader(self):
        """
        Return the validation data loader.
        
        Returns:
            DataLoader: Validation data loader.
        """
        return self.loader_test
    
    def test_dataloader(self):
        """
        Return the test data loader.
        
        Returns:
            DataLoader: Test data loader.
        """
        return self.loader_test
