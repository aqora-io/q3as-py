from pydantic import BaseModel


class EstimatorOptions(BaseModel):
    shots: int = 1024
