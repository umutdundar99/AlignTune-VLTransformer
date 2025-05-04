from torch.utils.data import Sampler
import random


class LengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length_to_indices = self._group_by_length()

    def _group_by_length(self):
        length_to_indices = {}
        for idx in range(len(self.dataset)):
            input_length = self.dataset.get_input_length(idx)
            if input_length not in length_to_indices:
                length_to_indices[input_length] = []
            length_to_indices[input_length].append(idx)
        return length_to_indices

    def __iter__(self):
        batches = []
        for indices in self.length_to_indices.values():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        random.shuffle(batches)
        for batch in batches:
            return iter(batch)

    def __len__(self):
        total = 0
        for indices in self.length_to_indices.values():
            total += len(indices) // self.batch_size
        return total
