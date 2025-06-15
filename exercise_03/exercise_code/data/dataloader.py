"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Determine the indices to iterate over
        indices = np.arange(len(self.dataset))                                                             #
        if self.shuffle :
        # Get a list of all numbers from 0 to n-1 in a random order 
            indices = np.random.permutation(indices)
        
        # Create batches using a generator
        batch = []
        for idx in indices :
        # Append the sample to the batch
            batch.append(self.dataset[idx])

            # If batch size is reached, yield the batch
            if len(batch) == self.batch_size :
                yield self.combine_batch_dicts(batch)
                batch = []

       # Yield the last batch if drop_last is False and batch is not empty
        if batch and not self.drop_last:
            yield self.combine_batch_dicts(batch)


    def __len__(self):
        num_samples = len(self.dataset)
        # The number of batches sample from the dataset
        if self.drop_last :
            length = num_samples // self.batch_size 
        else:
            length = (num_samples + self.batch_size - 1) // self.batch_size

        return length

    def combine_batch_dicts(self, batch):
        combined_batch = {key: np.array([d[key] for d in batch]) for key in batch[0].keys()}
        return combined_batch
