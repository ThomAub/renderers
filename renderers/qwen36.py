"""Qwen3.6 Renderer — mirrors the Qwen3.6 Jinja chat template.

Delta vs Qwen3.5 (template line 122):

- Tool-call argument serialization changed to ``tojson`` for every non-string
  value. Bools now render as ``true``/``false`` (not ``True``/``False``) and
  ``None`` as ``null`` (not ``None``), fixing the single-turn extension-break
  mode where a boolean parameter's case drifted across a re-render.

The template's other delta vs Qwen3.5 (a ``preserve_thinking`` toggle that
flips historical ``<think>`` retention on or off globally) is no longer
exposed as a constructor kwarg — its default-False behaviour matches
Qwen3.5 and is now baked in. Callers who want the toggled-on behaviour
pass ``preserve_all_thinking=True`` to ``create_renderer``, the
renderer-agnostic spelling of the same intent.

Everything else — tool system prompt, tool-call XML structure, thinking
markers, bridge logic, parser — is identical to Qwen3.5.
"""

from __future__ import annotations

import json
from typing import Any

from renderers._native_router import (
    load_native,
    native_enabled,
    resolve_tokenizer_path,
)
from renderers.qwen35 import Qwen35Renderer, _detect_enable_thinking_default


class Qwen36Renderer(Qwen35Renderer):
    """Deterministic message → token renderer for Qwen3.6 models."""

    def __new__(
        cls,
        tokenizer,
        *,
        processor=None,
        enable_thinking=None,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
        image_cache_max=256,
    ):
        # Route to native only for Qwen3.6 specifically — never fall
        # through to the parent's qwen35 router (the renderer flag is
        # different).
        if native_enabled("qwen36") and processor is None:
            native = load_native()
            if native is not None:
                if enable_thinking is None:
                    enable_thinking = _detect_enable_thinking_default(tokenizer)
                path = resolve_tokenizer_path(tokenizer)
                return native.Renderer.qwen36(
                    path,
                    enable_thinking=enable_thinking,
                    preserve_all_thinking=preserve_all_thinking,
                    preserve_thinking_between_tool_calls=preserve_thinking_between_tool_calls,
                )
        # Skip Qwen35Renderer.__new__ (would also try to route, with the
        # wrong flag). Go straight to object.
        return object.__new__(cls)

    @staticmethod
    def _render_arg_value(arg_value: Any) -> str:
        if isinstance(arg_value, str):
            return arg_value
        return json.dumps(arg_value, ensure_ascii=False)
