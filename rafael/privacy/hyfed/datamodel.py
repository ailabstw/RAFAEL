from pydantic import BaseModel
import numpy as np


class Noise(BaseModel):
    noise: list

    def toarray(self):
        return np.array(self.noise)

    def todict(self):
        return {"noise": self.noise}


class ModelParameters(BaseModel):
    parameters: list

    def toarray(self):
        return np.array(self.parameters)

    def todict(self):
        return {"parameters": self.parameters}
