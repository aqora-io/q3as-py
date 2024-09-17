from typing import Type, TypeVar
import json
from io import IOBase
from pydantic import BaseModel

from httpx import BasicAuth, URL, Headers, Request, Client as BaseClient, Response

from q3as.api import ApiClient, JobRequest, JobInfo


class Credentials:
    id: str
    secret: str

    def __init__(self, id: str, secret: str):
        self.id = id
        self.secret = secret

    @classmethod
    def load(cls, file: str | IOBase):
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


class Client(ApiClient):
    client: BaseClient
    req: RequestBuilder
    res: ResponseBuilder

    def __init__(self, credentials: Credentials, *, url: str = "https://q3as.aqora.io"):
        self.req = RequestBuilder(url)
        self.res = ResponseBuilder()
        self.client = BaseClient(auth=credentials.auth())

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def create_job(self, job: JobRequest) -> JobInfo:
        return self.res.parse(JobInfo, self.client.send(self.req.create_job(job)))
