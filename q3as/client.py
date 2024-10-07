from typing import Type, TypeVar, Optional, Tuple
from contextlib import AbstractContextManager
import json
import time
from io import IOBase
from pydantic import BaseModel

from httpx import (
    BasicAuth,
    URL,
    Headers,
    Request,
    Client as BaseClient,
    Response,
)

from q3as.api import JobRequest, JobInfo, JobStatus, Job, JobResult


class Credentials:
    """
    Credentials for authenticating with the server.
    """

    id: str
    secret: str

    def __init__(self, id: str, secret: str):
        """
        Create a new credentials instance.
        """
        self.id = id
        self.secret = secret

    @classmethod
    def load(cls, file: str | IOBase):
        """
        Load credentials from a file.
        """
        if isinstance(file, str):
            file = open(file)
        return cls(**json.load(file))

    def auth(self):
        return BasicAuth(self.id, self.secret)


class RequestBuilder:
    url: URL

    def __init__(self, url: str):
        self.url = URL(url)

    def build_request(
        self, method: str, path: str, headers: dict[str, str] = {}, **kwargs
    ) -> Request:
        return Request(
            method,
            self.url.copy_with(path=path),
            headers=Headers(
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    **headers,
                }
            ),
            **kwargs,
        )

    def create_job(self, job: JobRequest) -> Tuple[Type[JobInfo], Request]:
        return (
            JobInfo,
            self.build_request("POST", "/api/v1/jobs/", content=job.model_dump_json()),
        )

    def get_job_info(self, slug: str) -> Tuple[Type[JobInfo], Request]:
        return (JobInfo, self.build_request("GET", f"/api/v1/jobs/{slug}"))

    def get_job_request(self, slug: str) -> Tuple[Type[JobRequest], Request]:
        return (JobRequest, self.build_request("GET", f"/api/v1/jobs/{slug}/request"))

    def get_job_result(self, slug: str) -> Tuple[Type[JobResult], Request]:
        return (JobResult, self.build_request("GET", f"/api/v1/jobs/{slug}/result"))

    def pause_job(self, slug: str) -> Tuple[Type[JobInfo], Request]:
        return (JobInfo, self.build_request("PUT", f"/api/v1/jobs/{slug}/pause"))

    def resume_job(self, slug: str) -> Tuple[Type[JobInfo], Request]:
        return (JobInfo, self.build_request("PUT", f"/api/v1/jobs/{slug}/resume"))

    def delete_job(self, slug: str) -> Tuple[Type[JobInfo], Request]:
        return (JobInfo, self.build_request("DELETE", f"/api/v1/jobs/{slug}"))


class APIError(Exception):
    def __init__(self, message: str, response: Response):
        super().__init__(message)
        self.response = response


T = TypeVar("T")


class ResponseBuilder:
    def check(self, response: Response):
        if response.is_success:
            return
        try:
            message = response.json()["detail"]
        except Exception:
            message = response.text
        raise APIError(message, response)

    def parse[T: BaseModel](self, model: Type[T], response: Response) -> T:
        self.check(response)
        return model.model_validate(response.json())


JobInput = TypeVar("JobInput")


class Client(AbstractContextManager):
    """Synchronous client for creating and managing Jobs"""

    client: BaseClient
    req: RequestBuilder
    res: ResponseBuilder

    def __init__(self, credentials: Credentials, *, url: str = "https://q3as.aqora.io"):
        """Create a new client instance"""
        self.req = RequestBuilder(url)
        self.res = ResponseBuilder()
        self.client = BaseClient(auth=credentials.auth())

    def close(self):
        """
        Close the client
        """
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _send[T: BaseModel](self, request: Tuple[Type[T], Request]) -> T:
        return self.res.parse(request[0], self.client.send(request[1]))

    def _create_job(self, job: JobRequest) -> JobInfo:
        return self._send(self.req.create_job(job))

    def _get_job_info(self, slug: str) -> JobInfo:
        return self._send(self.req.get_job_info(slug))

    def _get_job_request(self, slug: str) -> JobRequest:
        return self._send(self.req.get_job_request(slug))

    def _get_job_result(self, slug: str) -> JobResult:
        return self._send(self.req.get_job_result(slug))

    def _pause_job(self, slug: str) -> JobInfo:
        return self._send(self.req.pause_job(slug))

    def _resume_job(self, slug: str) -> JobInfo:
        return self._send(self.req.resume_job(slug))

    def _delete_job(self, slug: str) -> JobInfo:
        return self._send(self.req.delete_job(slug))

    def _wait_for_job(
        self,
        name: str,
        polling_interval: float = 1.0,
        max_wait: Optional[float] = None,
    ) -> Tuple[JobInfo, JobResult]:
        started = time.time()
        while True:
            job = self._get_job_info(name)
            if job.status is not JobStatus.STARTED:
                result = self._get_job_result(name)
                if result.is_some():
                    return (job, result)
            if max_wait is not None and time.time() - started >= max_wait:
                raise TimeoutError("Job did not finish in time")
            time.sleep(polling_interval)

    def create_job(self, request: JobRequest) -> Job:
        """
        Create a new Job
        """
        info = self._create_job(request)
        modified_request = self._get_job_request(info.slug)
        return Job(self, info, modified_request)

    def get_job(self, name: str) -> Job:
        """
        Get a job by name
        """
        info = self._get_job_info(name)
        request = self._get_job_request(name)
        return Job(self, info, request)
