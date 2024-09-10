from typing import Any
import json
from io import IOBase
from aiohttp import ClientSession, BasicAuth, ClientResponse, ClientResponseError

from qaaas.api import VQE


class Credentials:
    id: str
    secret: str

    def __init__(self, id: str, secret: str):
        self.id = id
        self.secret = secret

    @classmethod
    def load(cls, file: IOBase):
        return cls(**json.load(file))

    def auth(self):
        return BasicAuth(self.id, self.secret)


class APIError(Exception):
    response: ClientResponseError
    reason: str

    def __init__(self, response: ClientResponseError, body: dict[str, Any]):
        self.response = response
        self.reason = body.get("detail", "Unknown")

    def __str__(self):
        return f"[{self.response.status} {self.response.message}]: {json.dumps(self.reason, indent=2)}"


class Client:
    session: ClientSession

    def __init__(
        self, credentials: Credentials, *, url: str = "https://qaaas.aqora.io"
    ):
        self.session = ClientSession(
            url,
            auth=credentials.auth(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    async def close(self):
        await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def _process_response(self, response: ClientResponse):
        json = await response.json()
        try:
            response.raise_for_status()
        except ClientResponseError as e:
            raise APIError(e, json)
        return await response.json()

    async def create_job(self, job: VQE):
        async with self.session.post(
            "/api/v1/jobs/", data=job.model_dump_json()
        ) as response:
            return await self._process_response(response)
