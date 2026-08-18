from __future__ import annotations

from typing import Any

from local_nms.ai import output_models


DEFAULT_MODEL = {"name": "gpt", "version": "5.6-luna", "thinking": "low"}
DEFAULT_MAX_COMPLETION_TOKENS = 16000
GPT_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}


def configured_model(model: dict[str, Any] | None) -> tuple[str, str]:
    if model is not None and not isinstance(model, dict):
        raise ValueError("nmsAi.model must be an object.")
    settings = {**DEFAULT_MODEL, **(model or {})}
    name = str(settings.get("name") or "").strip().lower()
    version = str(settings.get("version") or "").strip()
    if not name or not version:
        raise ValueError("nmsAi.model requires non-empty name and version values.")
    thinking = str(settings.get("thinking") or DEFAULT_MODEL["thinking"]).strip().lower()
    if name == "gpt":
        # Older QAQC configs used `minimal`, which newer GPT chat models reject.
        thinking = "low" if thinking == "minimal" else thinking
        if thinking not in GPT_REASONING_EFFORTS:
            supported = ", ".join(sorted(GPT_REASONING_EFFORTS))
            raise ValueError(f"Unsupported GPT reasoning effort {thinking!r}; use one of: {supported}.")
    return f"{name}-{version}", thinking


def _chat_model(model: dict[str, Any] | None, timeout: int, max_completion_tokens: int):
    model_name, thinking = configured_model(model)
    if model_name.lower().startswith("gpt-"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            reasoning_effort=thinking,
            temperature=0,
            timeout=timeout,
            max_completion_tokens=max_completion_tokens,
        )
    if model_name.lower().startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            thinking_level=thinking,
            temperature=0,
            request_timeout=timeout,
            max_tokens=max_completion_tokens,
        )
    raise ValueError(f"Unsupported nmsAi model: {model_name}")


def process_reasonings(
    system_prompt: str,
    user_prompt: str,
    *,
    model: dict[str, Any] | None = None,
    timeout: int = 60,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> output_models.NmsReasoningOutput:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as error:
        raise RuntimeError("Install the LangChain dependencies to enable NMS AI reasoning.") from error

    parser = output_models.reasoning_parser()
    response = _chat_model(model, timeout, max_completion_tokens).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{user_prompt}\n\n{parser.get_format_instructions()}"),
        ],
        config={"run_name": "nms-ai-reasoning"},
    )
    parsed = parser.invoke(response)
    return output_models.normalize_reasoning_response(parsed)
