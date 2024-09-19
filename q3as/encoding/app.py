from __future__ import annotations
from typing import Any

from pydantic import BaseModel

from q3as.app.application import Application, ApplicationName

import q3as.app.qubo


class EncodedApplication(BaseModel):
    name: ApplicationName
    data: Any

    @classmethod
    def encode(cls, app: Application) -> EncodedApplication:
        return cls(name=app.name(), data=app.encode())

    def decode(self) -> Application:
        if self.name == "qubo":
            return q3as.app.qubo.Qubo.decode(self.data)
        raise ValueError(f"Unknown application name: {self.name}")
