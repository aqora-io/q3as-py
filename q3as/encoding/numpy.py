from __future__ import annotations
from typing import Literal

import io
import base64
import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel


class EncodedArray(BaseModel):
    encoding: Literal["npy"] = "npy"
    base64: bytes

    @classmethod
    def encode(cls, array: ArrayLike) -> EncodedArray:
        data = io.BytesIO()
        np.save(data, array, allow_pickle=False)
        return cls(encoding="npy", base64=base64.b64encode(data.getvalue()))

    def decode(self) -> np.ndarray:
        data = base64.b64decode(self.base64)
        return np.load(io.BytesIO(data), allow_pickle=False)
