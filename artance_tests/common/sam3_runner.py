from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from common.paths import CAPX_ROOT, ensure_capx_on_path
from common.visualization import save_prompt_visualization


@dataclass(frozen=True)
class Sam3RunConfig:
    image: Path
    prompts: tuple[str, ...]
    output_dir: Path
    service_url: str
    top_k: int = 3
    show: bool = False


def load_sam3_client_module() -> Any:
    """Load cap-x SAM3 client without importing the heavier integrations package."""
    ensure_capx_on_path()
    sam3_path = CAPX_ROOT / "capx/integrations/vision/sam3.py"
    spec = importlib.util.spec_from_file_location("capx_sam3_client_for_artance_tests", sam3_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SAM3 client module from {sam3_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_sam3_service_url() -> str:
    return str(load_sam3_client_module().SERVICE_URL)


def load_rgb_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def run_sam3_text_prompts(
    image: Image.Image,
    prompts: tuple[str, ...],
    service_url: str,
) -> dict[str, list[dict[str, Any]]]:
    sam3_client = load_sam3_client_module()
    sam3_client.SERVICE_URL = service_url
    segment = sam3_client.init_sam3()
    return {prompt: segment(image, prompt) for prompt in prompts}


def save_sam3_run_outputs(
    image: Image.Image,
    cfg: Sam3RunConfig,
    prompt_results: dict[str, list[dict[str, Any]]],
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = cfg.image.stem
    summary: dict[str, Any] = {
        "config": {
            **asdict(cfg),
            "image": str(cfg.image),
            "output_dir": str(cfg.output_dir),
            "prompts": list(cfg.prompts),
        },
        "prompts": {},
    }

    for prompt, results in prompt_results.items():
        saved = save_prompt_visualization(
            image=image,
            prompt=prompt,
            results=results,
            output_dir=cfg.output_dir,
            image_stem=image_stem,
            top_k=cfg.top_k,
            show=cfg.show,
        )
        summary["prompts"][prompt] = {
            "num_results": len(results),
            "saved_results": saved,
        }

    summary_path = cfg.output_dir / image_stem / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def run_sam3_text_test(cfg: Sam3RunConfig) -> tuple[dict[str, list[dict[str, Any]]], Path]:
    image = load_rgb_image(cfg.image)
    prompt_results = run_sam3_text_prompts(image, cfg.prompts, cfg.service_url)
    summary_path = save_sam3_run_outputs(image, cfg, prompt_results)
    return prompt_results, summary_path
