from typing import Literal
from pydantic import BaseModel


class SamplerOptions(BaseModel):
    shots: int = 1024


class EstimatorOptions(BaseModel):
    shots: int = 1024


class RunOptions(BaseModel):
    backend: Literal["cpu_aer"] = "cpu_aer"
    sampler: SamplerOptions = SamplerOptions()
    estimator: EstimatorOptions = EstimatorOptions()
