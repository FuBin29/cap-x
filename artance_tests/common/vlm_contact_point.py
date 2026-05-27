from __future__ import annotations

import ast
import base64
import json
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from common.paths import ensure_capx_on_path


MAX_COORDINATE_VALUE = 1000
DEFAULT_VLM_SERVER_URL = "http://127.0.0.1:8110/chat/completions"
DEFAULT_VLM_MODEL = "gemini-2.5-pro"
DEFAULT_VLM_MAX_TOKENS = None


GENERIC_CONTACT_POINT_INSTRUCTION_TEMPLATE = """
You are a robotic vision and manipulation perception module.

Task:
{task_goal}

Contact Point Rules:
{contact_rules}

Coordinate System:
- Use normalized integer coordinates [{coordinate_names}].
- [0, 0] is the top-left corner of the image.
- [{max_coordinate}, {max_coordinate}] is the bottom-right corner of the image.

Constraint:
- Output ONLY one coordinate in this exact format: [{coordinate_names}]
- Do NOT include any descriptive text, reasoning, markdown, or additional characters.
""".strip()


@dataclass(frozen=True)
class VlmContactPointPromptSpec:
    task_goal: str
    contact_rules: tuple[str, ...]
    user_instruction: str = "Find the single best contact point for the task."


@dataclass(frozen=True)
class VlmContactPointConfig:
    image: Path
    output_dir: Path
    prompt_spec: VlmContactPointPromptSpec
    model: str = DEFAULT_VLM_MODEL
    server_url: str = DEFAULT_VLM_SERVER_URL
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = DEFAULT_VLM_MAX_TOKENS
    reasoning_effort: str = "low"
    prompt_order: str | None = None
    debug: bool = False


@dataclass(frozen=True)
class VlmContactPointResult:
    normalized_xy: tuple[int, int]
    pixel_xy: tuple[int, int]
    raw_coordinate: tuple[int, int]
    prompt_order: str
    raw_response: str
    image_size: tuple[int, int]


class VlmContactPointParseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str,
        prompt_order: str,
        model: str,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.prompt_order = prompt_order
        self.model = model


def is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def resolve_prompt_order(model: str, prompt_order: str | None) -> str:
    if prompt_order is None or prompt_order == "auto":
        return "yx" if is_gemini_model(model) else "xy"
    if prompt_order not in {"xy", "yx"}:
        raise ValueError(f"prompt_order must be one of auto, xy, yx; got {prompt_order!r}")
    return prompt_order


def render_contact_rules(contact_rules: tuple[str, ...]) -> str:
    if not contact_rules:
        raise ValueError("prompt_spec.contact_rules must contain at least one rule.")
    rendered = [f"- {rule.strip()}" for rule in contact_rules if rule.strip()]
    if not rendered:
        raise ValueError("prompt_spec.contact_rules must contain at least one non-empty rule.")
    return "\n".join(rendered)


def system_instruction_for_order(
    prompt_order: str,
    prompt_spec: VlmContactPointPromptSpec,
) -> str:
    if prompt_order == "yx":
        coordinate_names = "y, x"
    elif prompt_order == "xy":
        coordinate_names = "x, y"
    else:
        raise ValueError(f"Unsupported prompt order: {prompt_order}")

    return GENERIC_CONTACT_POINT_INSTRUCTION_TEMPLATE.format(
        task_goal=prompt_spec.task_goal.strip(),
        contact_rules=render_contact_rules(prompt_spec.contact_rules),
        coordinate_names=coordinate_names,
        max_coordinate=MAX_COORDINATE_VALUE,
    )


def load_rgb_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def encode_image_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_contact_point_prompt(
    image_path: Path,
    prompt_order: str,
    prompt_spec: VlmContactPointPromptSpec,
) -> list[dict[str, Any]]:
    instruction = system_instruction_for_order(prompt_order, prompt_spec)
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_spec.user_instruction,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_data_uri(image_path)},
                },
            ],
        },
    ]


def _extract_literal_candidate(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    match = re.search(r"\[[^\[\]]*-?\d+(?:\.\d+)?[^\[\]]*,[^\[\]]*-?\d+(?:\.\d+)?[^\[\]]*\]", cleaned)
    if match:
        return match.group(0)
    raise ValueError(f"Could not find a coordinate pair in VLM response: {text!r}")


def _first_coordinate_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        for key in ("point_2d", "point", "coordinate", "coordinates"):
            if key in value:
                return _first_coordinate_pair(value[key])
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            return int(round(value[0])), int(round(value[1]))
        for item in value:
            try:
                return _first_coordinate_pair(item)
            except ValueError:
                continue
    raise ValueError(f"Could not parse a coordinate pair from {value!r}")


def parse_contact_point_response(text: str, prompt_order: str) -> tuple[tuple[int, int], tuple[int, int]]:
    candidate = _extract_literal_candidate(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(candidate)

    raw_a, raw_b = _first_coordinate_pair(parsed)
    if not (0 <= raw_a <= MAX_COORDINATE_VALUE and 0 <= raw_b <= MAX_COORDINATE_VALUE):
        raise ValueError(
            f"Coordinate values must be within 0..{MAX_COORDINATE_VALUE}; got {(raw_a, raw_b)}"
        )

    if prompt_order == "yx":
        normalized_xy = (raw_b, raw_a)
    elif prompt_order == "xy":
        normalized_xy = (raw_a, raw_b)
    else:
        raise ValueError(f"Unsupported prompt order: {prompt_order}")
    return (raw_a, raw_b), normalized_xy


def normalized_to_pixel_xy(normalized_xy: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    x_norm, y_norm = normalized_xy
    x_px = round(x_norm / MAX_COORDINATE_VALUE * (width - 1))
    y_px = round(y_norm / MAX_COORDINATE_VALUE * (height - 1))
    return (
        min(max(x_px, 0), width - 1),
        min(max(y_px, 0), height - 1),
    )


def query_vlm_contact_point(cfg: VlmContactPointConfig) -> VlmContactPointResult:
    ensure_capx_on_path()
    from capx.llm.client import ModelQueryArgs, query_model

    image = load_rgb_image(cfg.image)
    prompt_order = resolve_prompt_order(cfg.model, cfg.prompt_order)
    prompt = build_contact_point_prompt(cfg.image, prompt_order, cfg.prompt_spec)
    args = ModelQueryArgs(
        model=cfg.model,
        server_url=cfg.server_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        reasoning_effort=cfg.reasoning_effort,
        debug=cfg.debug,
    )
    response = query_model(args, prompt)
    raw_response = response["content"] if isinstance(response, dict) else str(response)
    try:
        raw_coordinate, normalized_xy = parse_contact_point_response(raw_response, prompt_order)
    except ValueError as exc:
        raise VlmContactPointParseError(
            f"Failed to parse VLM contact point response: {exc}",
            raw_response=raw_response,
            prompt_order=prompt_order,
            model=cfg.model,
        ) from exc
    pixel_xy = normalized_to_pixel_xy(normalized_xy, image.size)
    return VlmContactPointResult(
        normalized_xy=normalized_xy,
        pixel_xy=pixel_xy,
        raw_coordinate=raw_coordinate,
        prompt_order=prompt_order,
        raw_response=raw_response,
        image_size=image.size,
    )


def save_vlm_contact_point_summary(
    cfg: VlmContactPointConfig,
    result: VlmContactPointResult,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / cfg.image.stem / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            **asdict(cfg),
            "image": str(cfg.image),
            "output_dir": str(cfg.output_dir),
        },
        "result": asdict(result),
        "system_instruction": system_instruction_for_order(result.prompt_order, cfg.prompt_spec),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path

def save_vlm_contact_point_failure_summary(
    cfg: VlmContactPointConfig,
    *,
    error: Exception,
    raw_response: str | None = None,
    prompt_order: str | None = None,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / cfg.image.stem / "failure_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_prompt_order = prompt_order or resolve_prompt_order(cfg.model, cfg.prompt_order)
    summary = {
        "ok": False,
        "config": {
            **asdict(cfg),
            "image": str(cfg.image),
            "output_dir": str(cfg.output_dir),
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "raw_response": raw_response,
        "prompt_order": resolved_prompt_order,
        "system_instruction": system_instruction_for_order(resolved_prompt_order, cfg.prompt_spec),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
