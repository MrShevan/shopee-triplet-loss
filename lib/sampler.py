import numpy as np
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        labels: list,
        n_classes: int,
        n_samples: int
    ):
        """
        Samples n_classes and within these classes samples n_samples. If not enough images in
        class, oversampling from previous class images.
        Returns batches of size n_classes * n_samples.

        Args:
            labels: labels for each row in dataset
            n_classes: num of sampled labels
            n_samples: num of samples for each sampled label
        """
        self.labels = labels
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])

        self.last_used_label_indices = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        """
        Iterate throw dataset
        """
        self.count = 0
        while self.count < self.n_dataset:
            # choose classes per batch
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)

            indices = []
            for class_ in classes:
                last_idx = self.last_used_label_indices[class_]
                samples = self.label_to_indices[class_][last_idx: last_idx + self.n_samples]

                # over_sampling
                n_least_samples = self.n_samples - len(samples)
                if n_least_samples > 0:
                    samples = np.concatenate([
                        samples,
                        np.random.choice(self.label_to_indices[class_], n_least_samples)
                    ])

                indices.extend(samples)

                self.last_used_label_indices[class_] += len(samples)
                if self.last_used_label_indices[class_] > len(self.label_to_indices[class_]):
                    # reset used labels
                    np.random.shuffle(self.label_to_indices[class_])
                    self.last_used_label_indices[class_] = 0

            self.count += self.batch_size
            yield indices

    def __len__(self):
        """
        Num of iterations throw dataset
        """
        return self.n_dataset // self.batch_size
