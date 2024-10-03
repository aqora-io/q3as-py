from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypeVar
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


class JobResult(BaseModel):
    result: Optional[EncodedVQEResult]
    error: Optional[str]

    def is_some(self):
        return self.result is not None or self.error is not None

    def as_vqe_result(self) -> Optional[VQEResult]:
        if self.error is not None:
            raise Exception(self.error)
        if self.result is None:
            return self.result
        return self.result.decode()


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
    status: JobStatus
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime]
    finished_at: Optional[datetime.datetime]


class BaseJob:
    """
    Base class for a job.
    """

    info: JobInfo
    request: JobRequest

    def __init__(self, info: JobInfo, request: JobRequest):
        self.info = info
        self.request = request

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
        return self.request.input.decode()

    @property
    def run_options(self) -> RunOptions:
        """
        The run options.
        """
        return self.request.run_options

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


class Job(BaseJob):
    client: Client

    def __init__(self, client: Client, info: JobInfo, request: JobRequest):
        super().__init__(info, request)
        self.client = client

    def result(
        self, polling_interval: float = 1.0, max_wait: Optional[float] = None
    ) -> VQEResult:
        """
        Wait for the job to finish and return result
        If the job is not finished, return None. If the Job finished with an error, raise an exception.
        """
        if self.status is not JobStatus.STARTED:
            vqe_result = self.result_now()
            if vqe_result is not None:
                return vqe_result
        info, result = self.client._wait_for_job(
            self.info.slug, polling_interval, max_wait
        )
        self.info = info
        vqe_result = result.as_vqe_result()
        if vqe_result is None:
            raise ValueError("Expected result to not be None")
        return vqe_result

    def result_now(self) -> Optional[VQEResult]:
        """
        Get the result of the job now without waiting for it to finish.
        If the job is not finished, return None. If the Job finished with an error, raise an exception.
        """
        self.client._get_job_result(self.info.slug).as_vqe_result()

    def refetch(self) -> Job:
        """
        Refetch the job from the server.
        """
        info = self.client._get_job_info(self.info.slug)
        self.info = info
        return self

    def pause(self) -> Job:
        """
        Pause the job.
        """
        info = self.client._pause_job(self.info.slug)
        self.info = info
        return self

    def resume(self) -> Job:
        """
        Resume the job.
        """
        info = self.client._resume_job(self.info.slug)
        self.info = info
        return self

    def delete(self) -> Job:
        """
        Delete the job.
        """
        info = self.client._delete_job(self.info.slug)
        self.info = info
        return self
