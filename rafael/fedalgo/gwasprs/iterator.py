import abc

import numpy as np


def get_end_index(curr_idx, step, n):
    if curr_idx + step < n:
        return curr_idx + step
    else:
        return n


class IndexIterator:
    def __init__(self, end_idx: int, start_idx: int = 0, step: int = 1):
        self.current_idx = start_idx
        self.end_idx = end_idx
        self.step = step

    def reset(self):
        self.current_idx = 0

    def __iter__(self):
        return self

    def is_end(self):
        return self.current_idx >= self.end_idx

    def increase_step(self, step=0):
        if step == 0:
            self.current_idx += self.step
        else:
            self.current_idx += step

    def get_step(self, step):
        end_idx = get_end_index(self.current_idx, step, self.end_idx)
        return slice(self.current_idx, end_idx, None)

    def __next__(self):
        if not self.is_end():
            slc = self.get_step(self.step)
            self.increase_step()
            return slc
        else:
            raise StopIteration


class FullIterator:
    def __init__(self):
        super().__init__()

    def __next__(self):
        return slice(None, None, None)

    def is_end(self):
        return False

    def reset(self):
        pass  # don't need to do anything


class NDIterator(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def __iter__(self):
        return self


class SNPIterator(NDIterator):
    def __init__(self, n_SNP, step: int = 1, sample_iterator=FullIterator()):
        super().__init__()
        self.iter = IndexIterator(n_SNP, 0, step)
        self.sample_iterator = sample_iterator

    def increase_step(self, step):
        """
        Usage:
            SNPIterator.increase_step()
                the next SNP idx starts from current_idx + step

        More explicit usage can be found in unittest.
        """
        slc = self.iter.get_step(step)
        range = np.s_[next(self.sample_iterator), slc]

        if isinstance(self.sample_iterator, FullIterator):
            self.iter.increase_step(step)

        elif self.sample_iterator.is_end():
            self.sample_iterator.reset()
            self.iter.increase_step(step)

        return range

    def samples(self, n_sample, step: int = 1):
        self.sample_iterator = IndexIterator(n_sample, 0, step)
        return self

    def __next__(self):
        if not self.is_end():
            return self.increase_step(self.iter.step)
        raise StopIteration

    def is_end(self):
        return self.iter.is_end()

    def reset(self):
        self.iter.reset()
        self.sample_iterator.reset()


class SampleIterator(NDIterator):
    def __init__(self, n_sample, step: int = 1, snp_iterator=FullIterator()):
        super().__init__()
        self.iter = IndexIterator(n_sample, 0, step)
        self.snp_iterator = snp_iterator

    def increase_step(self, step):
        """
        Usage:
            SampleIterator.increase_step()
                the next Sample idx starts from current_idx + step

        More explicit usage can be found in unittest.
        """
        slc = self.iter.get_step(step)
        range = np.s_[slc, next(self.snp_iterator)]

        if isinstance(self.snp_iterator, FullIterator):
            self.iter.increase_step(step)

        elif self.snp_iterator.is_end():
            self.snp_iterator.reset()
            self.iter.increase_step(step)

        return range

    def snps(self, n_SNP, step: int = 1):
        self.snp_iterator = IndexIterator(n_SNP, 0, step)
        return self

    def __next__(self):
        if not self.is_end():
            return self.increase_step(self.iter.step)
        raise StopIteration

    def is_end(self):
        return self.iter.is_end()

    def reset(self):
        self.iter.reset()
        self.snp_iterator.reset()
