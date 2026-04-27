from typing import List, Literal

from pydantic import BaseModel


ClassificationType = Literal["input_block", "task_and_response"]


class TaskDefinition(BaseModel):
    cluster: str
    labels: List[str]

    def __str__(self) -> str:
        return f"Cluster={self.cluster}, labels={self.labels})"
