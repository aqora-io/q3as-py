from pydantic import BaseModel


class SamplerOptions(BaseModel):
    shots: int = 1024


class EstimatorOptions(BaseModel):
    shots: int = 1024


class RunOptions(BaseModel):
    sampler: SamplerOptions = SamplerOptions()
    estimator: EstimatorOptions = EstimatorOptions()
