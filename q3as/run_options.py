from typing import Literal
from pydantic import BaseModel


class SamplerOptions(BaseModel):
    """
    Options for the sampler.
    """

    shots: int = 1024
    "Number of shots for the sampler."


class EstimatorOptions(BaseModel):
    """
    Options for the estimator.
    """

    shots: int = 1024
    "Number of shots for the estimator."


type BackendName = Literal[
    "auto",
    "q3as",
    "q3as:aer_simulation_cpu",
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
    """
    Options for  running in the cloud.
    """

    backend: BackendName = "auto"
    "Name of the backend to use. See `q3as.run_options.BackendName` for possible values."
    sampler: SamplerOptions = SamplerOptions()
    "Options for the sampler."
    estimator: EstimatorOptions = EstimatorOptions()
    "Options for the estimator."
