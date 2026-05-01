import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.base import BaseEnv


class ApiBase(ABC):
    """Base class for tool APIs.

    Guidelines:
    - Expose public functions via `functions()`.
    - Each function should have a Google-style docstring with Args/Returns.
    - `combined_doc()` returns a standardized aggregate doc for prompts.
    - If your API needs access to the environment, implement `set_env(env)`.

    Example:
    class GraspPlanApi(ApiBase):
        def functions(self) -> dict[str, Callable[..., Any]]:
            # optionally, we can expose only one of the functions if needed
            return {"grasp_plan": self.grasp_plan, "grasp_plan_sim": self.grasp_plan_sim}

        def grasp_plan(self, depth: np.ndarray, intrinsics: np.ndarray) -> list[dict]:
            \"""
            Plan parallel-jaw grasps from a depth map.

            Args:
            depth: (H, W) float32 meters.
            intrinsics: (3, 3) float32 pinhole intrinsics.

            Returns:
            List of dicts: {pose: (4,4) float32, width: float, score: float}.
            \"""
            ...

        def grasp_plan_sim(self, depth: np.ndarray, intrinsics: np.ndarray) -> list[dict]:
            ...
    """

    def __init__(self, env: BaseEnv) -> None:
        self._env = env
        self._webui_enabled: bool = False

    def enable_webui(self, enabled: bool = True) -> None:
        """Enable or disable web UI execution logging for this API instance."""
        self._webui_enabled = enabled

    def _log_step(
        self,
        tool_name: str,
        text: str,
        images: list[np.ndarray | Image.Image | str]
        | np.ndarray
        | Image.Image
        | str
        | None = None,
        highlight: bool = False,
    ) -> None:
        """Log an execution step if web UI mode is enabled.

        This is a thin wrapper around :func:`capx.utils.execution_logger.log_step`
        that no-ops when the web UI flag is off, keeping control API code free of
        conditional checks.
        """
        if not self._webui_enabled:
            return
        from capx.utils.execution_logger import log_step

        log_step(tool_name=tool_name, text=text, images=images, highlight=highlight)

    def _log_step_update(
        self,
        text: str | None = None,
        images: list[np.ndarray | Image.Image | str]
        | np.ndarray
        | Image.Image
        | str
        | None = None,
    ) -> None:
        """Update the last logged step if web UI mode is enabled."""
        if not self._webui_enabled:
            return
        from capx.utils.execution_logger import log_step_update

        log_step_update(text=text, images=images)

    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return mapping of public function name -> callable."""
        # 要求子类必须实现该方法。它返回一个字典，将 API 暴露的公共函数名映射到实际的 \
        # 可调用对象（Callable）上。这决定了哪些方法可以被 LLM 或外部调用。

    def combined_doc(self) -> str:
        """Aggregate function docs in a simple, consistent format.

        Format per function:
            name(signature)
              Summary: first line of function doc
              Doc: full function docstring (Google style recommended)
        """
        # 通过 Python 的 inspect 模块自动提取 functions() 中注册的所有函数的 \
        # 签名（参数）和文档字符串（Docstrings），将其拼接成一个标准化的文本格式。 \
        # 这是专门用来生成给 LLM 提示词 (Prompt) 的，能够让智能体知晓该工具 \
        # 有哪些方法以及如何使用。
        
        # we need to discuss this design further down the line
        lines: list[str] = []
        # lines: list[str] = [f"API: {self.__class__.__name__}", ""]
        for name, fn in self.functions().items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(…)"
            doc = inspect.getdoc(fn) or ""
            # first = doc.splitlines()[0] if doc else ""
            lines.append(f"{name}{sig}")
            # if first:
            #     lines.append(f"  Summary: {first}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
        return "\n".join(lines).strip()


_API_FACTORIES: dict[str, Callable[[], ApiBase]] = {}


def register_api(name: str, factory: Callable[[], ApiBase]) -> None:
    _API_FACTORIES[name] = factory


@lru_cache(maxsize=256)
def get_api(name: str) -> Callable[[BaseEnv], ApiBase]:
    if name not in _API_FACTORIES:
        raise KeyError(f"API '{name}' not registered")
    return _API_FACTORIES[name]


def list_apis() -> list[str]:
    return list(_API_FACTORIES.keys())
