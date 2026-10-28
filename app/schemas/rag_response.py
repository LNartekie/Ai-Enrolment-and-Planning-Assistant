import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


class RAGDecision(BaseModel):
    """
    Define the allowed structure of a
    document-grounded AI response.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    status: Literal[
        "answered",
        "insufficient_evidence",
    ]

    answer: str | None

    citations: list[str]

    @model_validator(mode="after")
    def validate_decision(self):

        if self.status == "answered":

            if not self.answer or not self.answer.strip():
                raise ValueError(
                    "An answered response must contain "
                    "a non-empty answer."
                )

            if not self.citations:
                raise ValueError(
                    "An answered response must "
                    "contain citations."
                )

            if (
                "the uploaded documents do not "
                "establish an answer"
            ) in self.answer.lower():
                raise ValueError(
                    "An answered response cannot "
                    "also claim insufficient evidence."
                )

            unsupported_answer_phrases = (
                "not explicitly detailed",
                "not specified in",
                "not addressed in",
                "not covered by",
                "does not provide information",
                "no information is provided",
                "cannot determine from",
            )

            answer_text = self.answer.casefold()

            if any(
                phrase in answer_text
                for phrase in unsupported_answer_phrases
            ):
                raise ValueError(
                    "An answered response indicates that "
                    "the requested information is not "
                    "established by the evidence."
                )

        elif self.status == "insufficient_evidence":

            if self.answer is not None:
                raise ValueError(
                    "An insufficient-evidence response "
                    "must have a null answer."
                )

            if self.citations:
                raise ValueError(
                    "An insufficient-evidence response "
                    "must not include citations."
                )

        for citation in self.citations:

            if not re.fullmatch(r"S[1-9][0-9]*", citation):
                raise ValueError(
                    f"Invalid citation format: {citation}"
                )

        if len(self.citations) != len(set(self.citations)):
            raise ValueError(
                "Duplicate citation identifiers "
                "are not allowed."
            )

        return self