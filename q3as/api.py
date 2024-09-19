from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union, Awaitable, TypeVar
from enum import Enum
import datetime
from uuid import UUID

from pydantic import BaseModel

from q3as.encoding import EncodedVQE, EncodedVQEResult
from q3as.run_options import RunOptions

T = TypeVar("T")

type MaybeAwaitable[T] = Union[T, Awaitable[T]]


class JobRequest(BaseModel):
    input: EncodedVQE
    run_options: RunOptions = RunOptions()

    def send(self, client: ApiClient) -> MaybeAwaitable[JobInfo]:
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


class ApiClient(ABC):
    @abstractmethod
    def create_job(self, job: JobRequest) -> MaybeAwaitable[JobInfo]:
        pass
