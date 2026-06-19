from typing import Literal

from pydantic import BaseModel


SourceRightsKind = Literal["owned", "licensed", "public_domain", "transformed", "verbatim"]


class SourceRights(BaseModel):
    kind: SourceRightsKind
    rights_cleared: bool
    notes: str

