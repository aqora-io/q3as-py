from __future__ import annotations
from q3as.algo.vqe import VQE as BaseVQE, VQEBuilder as BaseVQEBuilder
from q3as.api import (
    JobRequest,
    JobInfo,
    EstimatorOptions,
    ApiClient,
    MaybeAwaitable,
)
from q3as.encoding import EncodedVQE


class VQE(BaseVQE):
    @classmethod
    def builder(cls) -> VQEBuilder:
        return VQEBuilder()

    def send(
        self, client: ApiClient, estimator: EstimatorOptions = EstimatorOptions()
    ) -> MaybeAwaitable[JobInfo]:
        job = JobRequest(input=EncodedVQE.encode(self), estimator=estimator)
        return client.create_job(job)


class VQEBuilder(BaseVQEBuilder):
    def build(self):
        vqe = super().build()
        vqe.__class__ = VQE
        return vqe

    def send(
        self, client: ApiClient, estimator: EstimatorOptions = EstimatorOptions()
    ) -> MaybeAwaitable[JobInfo]:
        job = JobRequest(input=EncodedVQE.encode(self.build()), estimator=estimator)
        return client.create_job(job)
