from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, TypeVar
from enum import Enum
import datetime
from uuid import UUID

from pydantic import BaseModel

from q3as.encoding import EncodedVQE, EncodedVQEResult
from q3as.run_options import RunOptions

if TYPE_CHECKING:
    from q3as.client import Client
    from q3as.algo.vqe import VQE, VQEResult

T = TypeVar("T")


class JobRequest(BaseModel):
    input: EncodedVQE
    run_options: RunOptions = RunOptions()

    def send(self, client: Client) -> Job:
        return client.create_job(self)


class JobStatus(Enum):
    STARTED = 0
    PAUSED = 1
    SUCCESS = 2
    ERROR = 3


class JobInfo(BaseModel):
    id: UUID
    slug: str
    request: JobRequest
    status: JobStatus
    result: Optional[Union[EncodedVQEResult, str]]
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime]
    finished_at: Optional[datetime.datetime]


class BaseJob:
    info: JobInfo

    def __init__(self, info: JobInfo):
        self.info = info

    @property
    def name(self) -> str:
        return self.info.slug

    @property
    def status(self) -> JobStatus:
        return self.info.status

    @property
    def input(self) -> VQE:
        return self.info.request.input.decode()

    @property
    def run_options(self) -> RunOptions:
        return self.info.request.run_options

    @property
    def created_at(self) -> datetime.datetime:
        return self.info.created_at

    @property
    def updated_at(self) -> Optional[datetime.datetime]:
        return self.info.updated_at

    @property
    def finished_at(self) -> Optional[datetime.datetime]:
        return self.info.finished_at

    def result_now(self) -> Optional[VQEResult]:
        if self.info.result is None:
            return None
        if isinstance(self.info.result, str):
            raise Exception(self.info.result)
        return self.info.result.decode()


class Job(BaseJob):
    client: Client

    def __init__(self, client: Client, info: JobInfo):
        super().__init__(info)
        self.client = client

    def wait(
        self, polling_interval: float = 1.0, max_wait: Optional[float] = None
    ) -> Job:
        if self.status is not JobStatus.STARTED:
            return self
        job = self.client.wait_for_job(self.info.slug, polling_interval, max_wait)
        self.info = job.info
        return job

    def result(
        self, polling_interval: float = 1.0, max_wait: Optional[float] = None
    ) -> VQEResult:
        job = self.wait(polling_interval, max_wait)
        result_now = job.result_now()
        if result_now is None:
            raise ValueError("Job did not return a result")
        return result_now

    def refetch(self) -> Job:
        job = self.client.get_job(self.info.slug)
        self.info = job.info
        return job

    def pause(self) -> Job:
        job = self.client.pause_job(self.info.slug)
        self.info = job.info
        return job

    def resume(self) -> Job:
        job = self.client.resume_job(self.info.slug)
        self.info = job.info
        return job

    def delete(self) -> Job:
        job = self.client.delete_job(self.info.slug)
        self.info = job.info
        return job
