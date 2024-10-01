from typing import Type, TypeVar, Optional
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

from q3as.api import JobRequest, JobInfo, JobStatus, Job


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

    def create_job(self, job: JobRequest) -> Request:
        return self.build_request(
            "POST", "/api/v1/jobs/", content=job.model_dump_json()
        )

    def get_job(self, slug: str) -> Request:
        return self.build_request("GET", f"/api/v1/jobs/{slug}")

    def pause_job(self, slug: str) -> Request:
        return self.build_request("PUT", f"/api/v1/jobs/{slug}/pause")

    def resume_job(self, slug: str) -> Request:
        return self.build_request("PUT", f"/api/v1/jobs/{slug}/resume")

    def delete_job(self, slug: str) -> Request:
        return self.build_request("DELETE", f"/api/v1/jobs/{slug}")


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
            message = response.json()["details"]
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

    def _send_job(self, request: Request) -> Job:
        return Job(self, self.res.parse(JobInfo, self.client.send(request)))

    def create_job(self, job: JobRequest) -> Job:
        """
        Create a new job
        """
        return self._send_job(self.req.create_job(job))

    def get_job(self, name: str) -> Job:
        """
        Get a job by its name
        """
        return self._send_job(self.req.get_job(name))

    def pause_job(self, name: str) -> Job:
        """
        Pause a job
        """
        return self._send_job(self.req.pause_job(name))

    def resume_job(self, name: str) -> Job:
        """
        Resume a job
        """
        return self._send_job(self.req.resume_job(name))

    def delete_job(self, name: str) -> Job:
        """
        Delete a job
        """
        return self._send_job(self.req.delete_job(name))

    def wait_for_job(
        self,
        name: str,
        polling_interval: float = 1.0,
        max_wait: Optional[float] = None,
    ) -> Job:
        """
        Wait for a job to finish
        """
        started = time.time()
        while True:
            job = self.get_job(name)
            if job.status is not JobStatus.STARTED:
                return job
            if max_wait is not None and time.time() - started >= max_wait:
                raise TimeoutError("Job did not finish in time")
            time.sleep(polling_interval)
