from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from common.paths import ensure_capx_on_path
from common.vlm_contact_point import (
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_SERVER_URL,
    MAX_COORDINATE_VALUE,
    VlmContactPointPromptSpec,
    normalized_to_pixel_xy,
    render_contact_rules,
)


GROUNDING_ROLES = ("object", "interaction_part", "support_part", "context")


STRUCTURED_GROUNDING_INSTRUCTION_TEMPLATE = """
You are a robotic vision and manipulation perception module.

Task:
{task_goal}

Your job:
1. Identify the main target object involved in the task.
2. Identify the interaction part that the robot should directly touch.
3. Identify the support part that contains, supports, or is physically associated with the interaction part.
4. Predict one best contact point for the robot end-effector.
5. Generate hierarchical SAM3 text prompts for segmenting the object and its relevant parts.

Definitions:
- target_object: the main object being manipulated.
- interaction_part: the specific visible part that the robot should directly contact.
- support_part: the larger part that contains or supports the interaction_part.
- contact_pixel: the best point for robot contact, using normalized integer coordinates [y, x].
- sam3_prompts.object: prompts for the whole target object.
- sam3_prompts.interaction_part: prompts for the direct contact part.
- sam3_prompts.support_part: prompts for the larger related part.
- sam3_prompts.context: optional prompts for nearby relevant structures.

Contact Point Rules:
{contact_rules}

SAM3 Prompt Rules:
- Prompts should be short noun phrases.
- Prompts should be image-specific and task-relevant.
- Include 1 to 3 prompts for object, interaction_part, and support_part when visible.
- Use more specific prompts before generic prompts.
- Do not include action verbs such as open, close, push, pull, press, or move.
- Do not include coordinate descriptions in SAM3 prompts.
- Avoid prompts for irrelevant background objects unless they are useful context.

Coordinate System:
- Use normalized integer coordinates [y, x].
- [0, 0] is the top-left corner of the image.
- [{max_coordinate}, {max_coordinate}] is the bottom-right corner of the image.

Output Format:
Output ONLY a valid JSON object in the following schema:

{{
  "target_object": "string",
  "interaction_part": "string",
  "support_part": "string",
  "contact_pixel": [int, int],
  "sam3_prompts": {{
    "object": ["string"],
    "interaction_part": ["string"],
    "support_part": ["string"],
    "context": ["string"]
  }}
}}

Do NOT include any explanation, markdown, comments, or additional text.
""".strip()


@dataclass(frozen=True)
class VlmInteractionGroundingConfig:
    image: Path
    output_dir: Path
    prompt_spec: VlmContactPointPromptSpec
    model: str = DEFAULT_VLM_MODEL
    server_url: str = DEFAULT_VLM_SERVER_URL
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = DEFAULT_VLM_MAX_TOKENS
    reasoning_effort: str = "low"
    debug: bool = False


@dataclass(frozen=True)
class VlmInteractionGroundingResult:
    target_object: str
    interaction_part: str
    support_part: str
    contact_pixel_yx: tuple[int, int]
    normalized_xy: tuple[int, int]
    pixel_xy: tuple[int, int]
    sam3_prompts: dict[str, tuple[str, ...]]
    raw_response: str
    image_size: tuple[int, int]


class VlmInteractionGroundingParseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str,
        model: str,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.model = model


def load_rgb_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def encode_image_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def system_instruction_for_structured_grounding(prompt_spec: VlmContactPointPromptSpec) -> str:
    return STRUCTURED_GROUNDING_INSTRUCTION_TEMPLATE.format(
        task_goal=prompt_spec.task_goal.strip(),
        contact_rules=render_contact_rules(prompt_spec.contact_rules),
        max_coordinate=MAX_COORDINATE_VALUE,
    )


def build_structured_grounding_prompt(
    image_path: Path,
    prompt_spec: VlmContactPointPromptSpec,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_instruction_for_structured_grounding(prompt_spec)},
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


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    raise ValueError(f"Could not find a JSON object in VLM response: {text!r}")


def _as_nonempty_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _parse_contact_pixel_yx(value: Any) -> tuple[int, int]:
    if not (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(item, (int, float)) for item in value[:2])
    ):
        raise ValueError("contact_pixel must be a two-item numeric array [y, x].")
    y_norm = int(round(float(value[0])))
    x_norm = int(round(float(value[1])))
    if not (0 <= y_norm <= MAX_COORDINATE_VALUE and 0 <= x_norm <= MAX_COORDINATE_VALUE):
        raise ValueError(
            f"contact_pixel values must be within 0..{MAX_COORDINATE_VALUE}; got {(y_norm, x_norm)}"
        )
    return y_norm, x_norm


def _clean_prompt_list(value: Any, role: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"sam3_prompts.{role} must be a list of strings.")

    prompts: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        prompt = " ".join(item.strip().split())
        if not prompt:
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
    return tuple(prompts)


def parse_structured_grounding_response(
    text: str,
    *,
    image_size: tuple[int, int],
) -> VlmInteractionGroundingResult:
    payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("Structured grounding response must be a JSON object.")

    contact_pixel_yx = _parse_contact_pixel_yx(payload.get("contact_pixel"))
    normalized_xy = (contact_pixel_yx[1], contact_pixel_yx[0])
    pixel_xy = normalized_to_pixel_xy(normalized_xy, image_size)

    raw_prompts = payload.get("sam3_prompts", {})
    if not isinstance(raw_prompts, dict):
        raise ValueError("sam3_prompts must be a JSON object.")
    sam3_prompts = {role: _clean_prompt_list(raw_prompts.get(role), role) for role in GROUNDING_ROLES}

    if not sam3_prompts["interaction_part"]:
        interaction_part = _as_nonempty_string(payload.get("interaction_part"), "interaction_part")
        sam3_prompts["interaction_part"] = (interaction_part,)
    if not sam3_prompts["support_part"]:
        support_part = _as_nonempty_string(payload.get("support_part"), "support_part")
        sam3_prompts["support_part"] = (support_part,)
    if not sam3_prompts["object"]:
        target_object = _as_nonempty_string(payload.get("target_object"), "target_object")
        sam3_prompts["object"] = (target_object,)

    return VlmInteractionGroundingResult(
        target_object=_as_nonempty_string(payload.get("target_object"), "target_object"),
        interaction_part=_as_nonempty_string(payload.get("interaction_part"), "interaction_part"),
        support_part=_as_nonempty_string(payload.get("support_part"), "support_part"),
        contact_pixel_yx=contact_pixel_yx,
        normalized_xy=normalized_xy,
        pixel_xy=pixel_xy,
        sam3_prompts=sam3_prompts,
        raw_response=text,
        image_size=image_size,
    )


def query_vlm_interaction_grounding(
    cfg: VlmInteractionGroundingConfig,
) -> VlmInteractionGroundingResult:
    ensure_capx_on_path()
    from capx.llm.client import ModelQueryArgs, query_model

    image = load_rgb_image(cfg.image)
    prompt = build_structured_grounding_prompt(cfg.image, cfg.prompt_spec)
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
        return parse_structured_grounding_response(raw_response, image_size=image.size)
    except ValueError as exc:
        raise VlmInteractionGroundingParseError(
            f"Failed to parse VLM interaction grounding response: {exc}",
            raw_response=raw_response,
            model=cfg.model,
        ) from exc


def save_vlm_interaction_grounding_summary(
    cfg: VlmInteractionGroundingConfig,
    result: VlmInteractionGroundingResult,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / cfg.image.stem / "structured_grounding_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": True,
        "config": {
            **asdict(cfg),
            "image": str(cfg.image),
            "output_dir": str(cfg.output_dir),
        },
        "result": asdict(result),
        "system_instruction": system_instruction_for_structured_grounding(cfg.prompt_spec),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def save_vlm_interaction_grounding_failure_summary(
    cfg: VlmInteractionGroundingConfig,
    *,
    error: Exception,
    raw_response: str | None = None,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / cfg.image.stem / "structured_grounding_failure_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
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
        "system_instruction": system_instruction_for_structured_grounding(cfg.prompt_spec),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def load_vlm_interaction_grounding_summary(summary_path: Path) -> VlmInteractionGroundingResult:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    result = data.get("result", data)
    if not isinstance(result, dict):
        raise ValueError(f"Could not find structured grounding result in {summary_path}")

    prompts = result.get("sam3_prompts", {})
    if not isinstance(prompts, dict):
        raise ValueError(f"sam3_prompts must be a dict in {summary_path}")

    return VlmInteractionGroundingResult(
        target_object=_as_nonempty_string(result.get("target_object"), "target_object"),
        interaction_part=_as_nonempty_string(result.get("interaction_part"), "interaction_part"),
        support_part=_as_nonempty_string(result.get("support_part"), "support_part"),
        contact_pixel_yx=tuple(int(v) for v in result["contact_pixel_yx"]),
        normalized_xy=tuple(int(v) for v in result["normalized_xy"]),
        pixel_xy=tuple(int(v) for v in result["pixel_xy"]),
        sam3_prompts={role: _clean_prompt_list(prompts.get(role), role) for role in GROUNDING_ROLES},
        raw_response=str(result.get("raw_response", "")),
        image_size=tuple(int(v) for v in result["image_size"]),
    )
