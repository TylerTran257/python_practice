import httpx
from httpx import NetworkError, TimeoutException


class GenerationServiceError(Exception):
    pass


class GenerationService:
    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:8080/v1"
        self.endpoint = "/chat/completions"
        self.timeout = 120.0
        self.temperature = 0.2
        self.max_output_tokens = 300
        self.max_context_chars = 6000
        self.max_chars_per_chunk = 1800

    def answer_question(self, question: str, sources: list[dict]) -> str:
        if not question.strip():
            raise GenerationServiceError("Question must not be empty")

        if len(sources) == 0:
            raise GenerationServiceError("No sources were provided for generation")

        messages = [
            {"role": "system", "content": self._build_system_message()},
            {"role": "user", "content": self._build_user_message(question, sources)},
        ]
        payload = self._create_payload(messages)
        data = self._post_chat_completion(payload)
        answer = self._parse_response(data)

        return answer

    def _build_system_message(self) -> str:
        """return the fixed grounding rules"""

        return """
            You answer questions using only the provided context.
            If the answer is not supported by the context, say that clearly.
            Do not invent facts.
            Keep the answer concise.
        """.strip()

    def _build_user_message(self, question: str, sources: list[dict]) -> str:
        """build the full user-visible prompt body"""

        formatted_context = self._format_sources_for_context(sources)

        return f"""
            Question:
            {question}

            Context:
            {formatted_context}
        """.strip()

    def _format_sources_for_context(self, sources: list[dict]) -> str:
        """
        apply the char budget across all retrieved chunks
        format chunks in retrieval order
        """

        blocks = []
        total_chars = 0

        for index, source in enumerate(sources, start=1):
            raw_text = source.get("text", "")
            truncated_text = self._truncate_source_text(
                raw_text, self.max_chars_per_chunk
            )

            header = (
                f"[Source {index} | "
                f"file={source.get('original_filename', 'unknown')} | "
                f"chunk={source.get('chunk_index', '?')} | "
                f"score={source.get('score', '?')}]"
            )

            block = f"{header}\n{truncated_text}".strip()

            if total_chars + len(block) > self.max_context_chars:
                break

            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)

    def _truncate_source_text(self, text: str, max_chars: int) -> str:
        """
        cap one chunk so a single source cannot dominate the prompt
        """

        cleaned_text = text.strip()
        if len(cleaned_text) <= max_chars:
            return cleaned_text

        return cleaned_text[:max_chars].rstrip() + "..."

    def _create_payload(self, messages: list[dict[str, str]]) -> dict:
        return {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }

    def _post_chat_completion(self, payload: dict) -> dict:
        """
        make the HTTP request to llama-server
        translate transport/upstream failures into one clean service error

        useful:
        - exception catching: what failures do I want to present to my app?
        except Exception as exc:
            print(type(exc), repr(exc))
            raise
        """

        url = self.base_url + self.endpoint

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except TimeoutException as exc:
            raise GenerationServiceError("Generation request timed out") from exc
        except NetworkError as exc:
            raise GenerationServiceError("Generation service is unavailable") from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationServiceError(
                f"Generation service returned HTTP {exc.response.status_code}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise GenerationServiceError(
                "Generation service returned invalid JSON"
            ) from exc

    def _parse_response(self, response_json: dict) -> str:
        choices = response_json.get("choices")
        if not choices:
            raise GenerationServiceError("Generation response was malformed")

        first_choice = choices[0]
        message = first_choice.get("message")
        if not message:
            raise GenerationServiceError("Generation response was malformed")

        content = message.get("content")
        if not content or not content.strip():
            raise GenerationServiceError("Generation response was empty")

        return content.strip()
