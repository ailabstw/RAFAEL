from abc import ABC

from . import rand

# HyFed mechanism

class HyFedRole(ABC):

    def __init__(self):
        pass


class HyFedClient(HyFedRole):

    def __init__(self):
        super().__init__()

    def update_noise(self, n: int):
        self._noise = rand.randint64(n)

    def generate_secrets(self, data):
        return self._noise, data + self._noise


class HyFedCompensator(HyFedRole):

    def __init__(self):
        super().__init__()

    def aggregate(self, *noises):
        return sum(noises)


class HyFedServer(HyFedRole):

    def __init__(self):
        super().__init__()

    def aggregate(self, aggregated_noise, *data):
        return sum(data) - aggregated_noise

    def update_model(self, model, aggregated_model):
        return model + aggregated_model
