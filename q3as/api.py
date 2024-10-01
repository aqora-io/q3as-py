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
    """
    Request to run a VQE job.
    """

    input: EncodedVQE
    "The input VQE to run."
    run_options: RunOptions = RunOptions()
    "The options to run the VQE with."

    def send(self, client: Client) -> Job:
        """
        Send the request to the server and create a job.
        """
        return client.create_job(self)


class JobStatus(Enum):
    """
    Enum representing the status of a job.
    """

    STARTED = 0
    "The job is started."
    PAUSED = 1
    "The job is paused."
    SUCCESS = 2
    "The job finished successfully."
    ERROR = 3
    "The job finished with an error."


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
    """
    Base class for a job.
    """

    info: JobInfo

    def __init__(self, info: JobInfo):
        self.info = info

    @property
    def name(self) -> str:
        """
        The name of the job.
        """
        return self.info.slug

    @property
    def status(self) -> JobStatus:
        """
        The status of the job.
        """
        return self.info.status

    @property
    def input(self) -> VQE:
        """
        The input VQE.
        """
        return self.info.request.input.decode()

    @property
    def run_options(self) -> RunOptions:
        """
        The run options.
        """
        return self.info.request.run_options

    @property
    def created_at(self) -> datetime.datetime:
        """
        The time the job was created.
        """
        return self.info.created_at

    @property
    def updated_at(self) -> Optional[datetime.datetime]:
        """
        The time the job was last updated.
        """
        return self.info.updated_at

    @property
    def finished_at(self) -> Optional[datetime.datetime]:
        """
        The time the job finished.
        """
        return self.info.finished_at

    def result_now(self) -> Optional[VQEResult]:
        """
        Get the result of the job without waiting for it to finish. May return None if the job is not finished, or raise an exception if the job failed.
        """
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
        """
        Wait for the job to finish
        """
        if self.status is not JobStatus.STARTED:
            return self
        job = self.client.wait_for_job(self.info.slug, polling_interval, max_wait)
        self.info = job.info
        return job

    def result(
        self, polling_interval: float = 1.0, max_wait: Optional[float] = None
    ) -> VQEResult:
        """
        Get the result of the job, waiting for it to finish if necessary.
        """
        job = self.wait(polling_interval, max_wait)
        result_now = job.result_now()
        if result_now is None:
            raise ValueError("Job did not return a result")
        return result_now

    def refetch(self) -> Job:
        """
        Refetch the job from the server.
        """
        job = self.client.get_job(self.info.slug)
        self.info = job.info
        return job

    def pause(self) -> Job:
        """
        Pause the job.
        """
        job = self.client.pause_job(self.info.slug)
        self.info = job.info
        return job

    def resume(self) -> Job:
        """
        Resume the job.
        """
        job = self.client.resume_job(self.info.slug)
        self.info = job.info
        return job

    def delete(self) -> Job:
        """
        Delete the job.
        """
        job = self.client.delete_job(self.info.slug)
        self.info = job.info
        return job
