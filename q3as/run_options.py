from typing import Literal
from pydantic import BaseModel


class SamplerOptions(BaseModel):
    shots: int = 1024


class EstimatorOptions(BaseModel):
    shots: int = 1024


type BackendName = Literal[
    "auto",
    "q3as",
    "q3as:aer_cpu",
    "scaleway",
    "scaleway:aer_simulation_2l4",
    "scaleway:aer_simulation_2l40s",
    "scaleway:aer_simulation_4l40s",
    "scaleway:aer_simulation_8l40s",
    "scaleway:aer_simulation_h100",
    "scaleway:aer_simulation_2h100",
    "scaleway:aer_simulation_pop_c16m128",
    "scaleway:aer_simulation_pop_c32m256",
    "scaleway:aer_simulation_pop_c64m512",
]


class RunOptions(BaseModel):
    backend: BackendName = "auto"
    sampler: SamplerOptions = SamplerOptions()
    estimator: EstimatorOptions = EstimatorOptions()
