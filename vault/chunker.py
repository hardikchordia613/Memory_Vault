"""Context-aware, dependency-light chunking for codebase ingestion.

``chunk_file`` is the public entry point.  It dispatches by file extension and
always returns embedding-ready text together with useful source metadata.
Tree-sitter and PyYAML are imported lazily so callers that only ingest plain
text or Markdown do not pay their import/startup cost.
"""

from __future__ import annotations

import importlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Chunk = dict[str, Any]

DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150
SMALL_CLASS_LIMIT = 1_000
SMALL_CONFIG_LIMIT = 1_500

MARKDOWN_EXTENSIONS = {".md", ".mdx"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class _LanguageSpec:
    module: str
    factory: str
    class_nodes: frozenset[str]
    function_nodes: frozenset[str]
    method_nodes: frozenset[str]


_COMMON_JS_CLASSES = frozenset({"class_declaration", "class_expression"})
_COMMON_JS_FUNCTIONS = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "function_expression",
        "generator_function",
        "arrow_function",
    }
)

LANGUAGES: dict[str, tuple[str, _LanguageSpec]] = {
    ".py": (
        "python",
        _LanguageSpec(
            "tree_sitter_python",
            "language",
            frozenset({"class_definition"}),
            frozenset({"function_definition"}),
            frozenset(),
        ),
    ),
    ".js": (
        "javascript",
        _LanguageSpec(
            "tree_sitter_javascript",
            "language",
            _COMMON_JS_CLASSES,
            _COMMON_JS_FUNCTIONS,
            frozenset({"method_definition"}),
        ),
    ),
    ".jsx": (
        "javascript",
        _LanguageSpec(
            "tree_sitter_javascript",
            "language",
            _COMMON_JS_CLASSES,
            _COMMON_JS_FUNCTIONS,
            frozenset({"method_definition"}),
        ),
    ),
    ".ts": (
        "typescript",
        _LanguageSpec(
            "tree_sitter_typescript",
            "language_typescript",
            _COMMON_JS_CLASSES | {"abstract_class_declaration"},
            _COMMON_JS_FUNCTIONS,
            frozenset(
                {"method_definition", "method_signature", "abstract_method_signature"}
            ),
        ),
    ),
    ".tsx": (
        "tsx",
        _LanguageSpec(
            "tree_sitter_typescript",
            "language_tsx",
            _COMMON_JS_CLASSES | {"abstract_class_declaration"},
            _COMMON_JS_FUNCTIONS,
            frozenset(
                {"method_definition", "method_signature", "abstract_method_signature"}
            ),
        ),
    ),
    ".java": (
        "java",
        _LanguageSpec(
            "tree_sitter_java",
            "language",
            frozenset(
                {
                    "class_declaration",
                    "interface_declaration",
                    "enum_declaration",
                    "record_declaration",
                }
            ),
            frozenset(),
            frozenset({"method_declaration", "constructor_declaration"}),
        ),
    ),
    ".go": (
        "go",
        _LanguageSpec(
            "tree_sitter_go",
            "language",
            frozenset(),
            frozenset({"function_declaration"}),
            frozenset({"method_declaration"}),
        ),
    ),
    ".rs": (
        "rust",
        _LanguageSpec(
            "tree_sitter_rust",
            "language",
            frozenset({"impl_item", "trait_item"}),
            frozenset({"function_item"}),
            frozenset(),
        ),
    ),
}

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def _display_path(file_path: str) -> str:
    """Use a compact filename in payloads while retaining the path in metadata."""

    return Path(file_path).name or file_path


def _breadcrumb(file_path: str, parts: Iterable[str] = ()) -> str:
    return " > ".join((f"File: {_display_path(file_path)}", *parts))


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _end_line_for_span(text: str, start: int, end: int) -> int:
    if end <= start:
        return _line_for_offset(text, start)
    return _line_for_offset(text, end - 1)


def _split_spans(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[int, int]]:
    """Return recursive-separator character spans with approximate overlap.

    Boundaries are chosen in priority order: blank line, line break, then a
    space.  A hard character boundary is used only if none occurs in the
    latter half of the desired window.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end
        if hard_end < text_length:
            minimum = start + max(overlap + 1, chunk_size // 2)
            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(separator, minimum, hard_end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break

        if end <= start:  # Defensive guard against malformed boundary logic.
            end = min(start + chunk_size, text_length)
        spans.append((start, end))
        if end == text_length:
            break
        start = max(start + 1, end - overlap)

    return spans


class ContextAwareChunker:
    """Routes source text to AST, Markdown, config, or text chunkers."""

    def __init__(
        self,
        *,
        small_class_limit: int = SMALL_CLASS_LIMIT,
        small_config_limit: int = SMALL_CONFIG_LIMIT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if small_class_limit <= 0 or small_config_limit <= 0:
            raise ValueError("size limits must be positive")
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        self.small_class_limit = small_class_limit
        self.small_config_limit = small_config_limit
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._parsers: dict[str, Any] = {}
        self._parser_lock = threading.RLock()

    def chunk_file(self, file_path: str, raw_text: str) -> list[Chunk]:
        """Chunk ``raw_text`` according to the extension in ``file_path``."""

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")
        if not raw_text.strip():
            return []

        extension = Path(file_path).suffix.lower()
        if extension in LANGUAGES:
            return self._chunk_code(file_path, raw_text, extension)
        if extension in MARKDOWN_EXTENSIONS:
            return self._chunk_markdown(file_path, raw_text)
        if extension in CONFIG_EXTENSIONS:
            return self._chunk_config(file_path, raw_text, extension)
        return self._chunk_unstructured(file_path, raw_text)

    def _get_parser(self, extension: str) -> Any:
        language_name, spec = LANGUAGES[extension]
        cache_key = extension if extension in {".ts", ".tsx"} else language_name
        with self._parser_lock:
            if cache_key in self._parsers:
                return self._parsers[cache_key]
            try:
                from tree_sitter import Language, Parser

                binding = importlib.import_module(spec.module)
            except ImportError as exc:
                package = spec.module.replace("_", "-")
                raise RuntimeError(
                    f"Cannot chunk {extension} code: install tree-sitter and {package}"
                ) from exc

            language = Language(getattr(binding, spec.factory)())
            parser = Parser(language)
            self._parsers[cache_key] = parser
            return parser

    @staticmethod
    def _node_text(node: Any, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8")

    @staticmethod
    def _point_row(point: Any) -> int:
        return point.row if hasattr(point, "row") else point[0]

    def _expanded_node(self, node: Any) -> Any:
        """Include Python decorators in a class/function chunk."""

        parent = node.parent
        if parent is not None and parent.type == "decorated_definition":
            return parent
        return node

    def _node_name(self, node: Any, source: bytes) -> str | None:
        for field in ("name", "type", "declarator"):
            child = node.child_by_field_name(field)
            if child is not None:
                value = self._node_text(child, source).strip()
                if value:
                    return value

        # JS/TS function expressions and arrows are commonly named by their
        # containing variable/property rather than by the function node.
        parent = node.parent
        while parent is not None and parent.type in {
            "parenthesized_expression",
            "variable_declarator",
            "pair",
            "public_field_definition",
        }:
            for field in ("name", "key"):
                child = parent.child_by_field_name(field)
                if child is not None:
                    value = self._node_text(child, source).strip()
                    if value:
                        return value
            parent = parent.parent
        return None

    @staticmethod
    def _class_label(node_type: str) -> str:
        if "interface" in node_type:
            return "Interface"
        if "trait" in node_type:
            return "Trait"
        if "impl" in node_type:
            return "Impl"
        if "record" in node_type:
            return "Record"
        if "enum" in node_type:
            return "Enum"
        return "Class"

    def _code_chunk(
        self,
        *,
        file_path: str,
        source: bytes,
        node: Any,
        breadcrumb_parts: tuple[str, ...],
        chunk_type: str,
        language: str,
        name: str | None,
        has_syntax_errors: bool,
    ) -> Chunk:
        selected = self._expanded_node(node)
        code = self._node_text(selected, source).strip()
        start_line = self._point_row(selected.start_point) + 1
        end_line = self._point_row(selected.end_point) + 1
        return {
            "text": f"{_breadcrumb(file_path, breadcrumb_parts)}\n\n{code}",
            "metadata": {
                "file_path": file_path,
                "chunk_type": chunk_type,
                "language": language,
                "name": name,
                "start_line": start_line,
                "end_line": end_line,
                "start_byte": selected.start_byte,
                "end_byte": selected.end_byte,
                "has_syntax_errors": has_syntax_errors,
            },
        }

    def _chunk_code(self, file_path: str, raw_text: str, extension: str) -> list[Chunk]:
        language, spec = LANGUAGES[extension]
        parser = self._get_parser(extension)
        source = raw_text.encode("utf-8")
        with self._parser_lock:
            tree = parser.parse(source)
        if tree is None:
            raise RuntimeError(f"tree-sitter failed to parse {file_path}")

        has_errors = bool(tree.root_node.has_error)
        chunks: list[Chunk] = []

        def walk(node: Any, scopes: tuple[str, ...]) -> None:
            if node.type in spec.class_nodes:
                name = self._node_name(node, source) or "<anonymous>"
                label = self._class_label(node.type)
                class_scopes = (*scopes, f"{label}: {name}")
                selected = self._expanded_node(node)
                if len(self._node_text(selected, source)) <= self.small_class_limit:
                    chunks.append(
                        self._code_chunk(
                            file_path=file_path,
                            source=source,
                            node=node,
                            breadcrumb_parts=class_scopes,
                            chunk_type=label.lower(),
                            language=language,
                            name=name,
                            has_syntax_errors=has_errors,
                        )
                    )
                    return

                before = len(chunks)
                for child in node.named_children:
                    walk(child, class_scopes)
                # Data-only classes/structural declarations must not disappear.
                if len(chunks) == before:
                    chunks.append(
                        self._code_chunk(
                            file_path=file_path,
                            source=source,
                            node=node,
                            breadcrumb_parts=class_scopes,
                            chunk_type=label.lower(),
                            language=language,
                            name=name,
                            has_syntax_errors=has_errors,
                        )
                    )
                return

            if node.type in spec.method_nodes or node.type in spec.function_nodes:
                name = self._node_name(node, source)
                is_method = node.type in spec.method_nodes or bool(scopes)
                label = "Method" if is_method else "Function"
                # Anonymous callbacks have little standalone retrieval value and
                # remain embedded in their enclosing named semantic block.
                if name is not None:
                    chunks.append(
                        self._code_chunk(
                            file_path=file_path,
                            source=source,
                            node=node,
                            breadcrumb_parts=(*scopes, f"{label}: {name}"),
                            chunk_type=label.lower(),
                            language=language,
                            name=name,
                            has_syntax_errors=has_errors,
                        )
                    )
                    return

            for child in node.named_children:
                walk(child, scopes)

        walk(tree.root_node, ())
        if chunks:
            return chunks

        # Valid code can contain only declarations/imports; keep it retrievable.
        return self._chunk_unstructured(file_path, raw_text, chunk_type="code_file")

    def _chunk_markdown(self, file_path: str, raw_text: str) -> list[Chunk]:
        lines = raw_text.splitlines(keepends=True)
        chunks: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []
        section_lines: list[str] = []
        section_offset = 0
        offset = 0
        fence: str | None = None

        def flush() -> None:
            if not section_lines:
                return
            content = "".join(section_lines)
            if not content.strip():
                return
            path = tuple(title for _, title in heading_stack)
            chunks.extend(
                self._split_with_settings(
                    file_path,
                    content,
                    path,
                    "markdown_section" if path else "markdown_preamble",
                    base_offset=section_offset,
                    base_line=_line_for_offset(raw_text, section_offset),
                    extra_metadata={"heading_path": list(path)},
                )
            )

        for line in lines:
            stripped = line.rstrip("\r\n")
            fence_match = _FENCE_RE.match(stripped)
            if fence_match:
                marker = fence_match.group(1)
                marker_char = marker[0]
                if fence is None:
                    fence = marker_char
                elif fence == marker_char:
                    fence = None

            heading = _HEADING_RE.match(stripped) if fence is None else None
            if heading:
                flush()
                section_lines.clear()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                section_offset = offset + len(line)
            else:
                if not section_lines:
                    section_offset = offset
                section_lines.append(line)
            offset += len(line)

        flush()
        return chunks

    def _serialize_config(self, value: Any, extension: str) -> str:
        if extension == ".json":
            return json.dumps(value, ensure_ascii=False, indent=2)
        try:
            yaml = importlib.import_module("yaml")
        except ImportError as exc:
            raise RuntimeError("Cannot chunk YAML: install PyYAML") from exc
        return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).rstrip()

    def _chunk_config(
        self, file_path: str, raw_text: str, extension: str
    ) -> list[Chunk]:
        if extension == ".json":
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc
        else:
            try:
                yaml = importlib.import_module("yaml")
            except ImportError as exc:
                raise RuntimeError("Cannot chunk YAML: install PyYAML") from exc
            try:
                parsed = yaml.safe_load(raw_text)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML in {file_path}: {exc}") from exc

        serialized = self._serialize_config(parsed, extension)
        total_lines = max(1, raw_text.count("\n") + 1)
        if len(serialized) <= self.small_config_limit:
            return [
                {
                    "text": f"{_breadcrumb(file_path)}\n\n{serialized}",
                    "metadata": {
                        "file_path": file_path,
                        "chunk_type": "config",
                        "key_path": None,
                        "start_line": 1,
                        "end_line": total_lines,
                    },
                }
            ]

        chunks: list[Chunk] = []

        def visit(value: Any, path: tuple[str, ...]) -> None:
            rendered = self._serialize_config(value, extension)
            if path and len(rendered) <= self.small_config_limit:
                key_path = ".".join(path)
                chunks.append(
                    {
                        "text": (
                            f"{_breadcrumb(file_path, (f'Key: {key_path}',))}"
                            f"\n\n{rendered}"
                        ),
                        "metadata": {
                            "file_path": file_path,
                            "chunk_type": "config_key",
                            "key_path": key_path,
                            "start_line": 1,
                            "end_line": total_lines,
                        },
                    }
                )
                return
            if isinstance(value, dict) and value:
                for key, child in value.items():
                    visit(child, (*path, str(key)))
                return
            if isinstance(value, list) and value:
                for index, child in enumerate(value):
                    visit(child, (*path, f"[{index}]"))
                return

            key_path = ".".join(path) if path else "$"
            chunks.append(
                {
                    "text": (
                        f"{_breadcrumb(file_path, (f'Key: {key_path}',))}"
                        f"\n\n{rendered}"
                    ),
                    "metadata": {
                        "file_path": file_path,
                        "chunk_type": "config_key",
                        "key_path": key_path,
                        "start_line": 1,
                        "end_line": total_lines,
                    },
                }
            )

        visit(parsed, ())
        return chunks

    def _split_with_settings(
        self,
        file_path: str,
        text: str,
        parts: Iterable[str],
        chunk_type: str,
        *,
        base_offset: int = 0,
        base_line: int = 1,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        prefix = _breadcrumb(file_path, parts)
        metadata_extra = extra_metadata or {}
        chunks: list[Chunk] = []
        for start, end in _split_spans(text, self.chunk_size, self.chunk_overlap):
            content = text[start:end].strip()
            if not content:
                continue
            chunks.append(
                {
                    "text": f"{prefix}\n\n{content}",
                    "metadata": {
                        "file_path": file_path,
                        "chunk_type": chunk_type,
                        "start_line": base_line + _line_for_offset(text, start) - 1,
                        "end_line": base_line + _end_line_for_span(text, start, end) - 1,
                        "start_char": base_offset + start,
                        "end_char": base_offset + end,
                        **metadata_extra,
                    },
                }
            )
        return chunks

    def _chunk_unstructured(
        self, file_path: str, raw_text: str, *, chunk_type: str = "text"
    ) -> list[Chunk]:
        return self._split_with_settings(file_path, raw_text, (), chunk_type)


_default_chunker = ContextAwareChunker()


def chunk_file(file_path: str, raw_text: str) -> list[Chunk]:
    """Return context-aware chunks for one file.

    This is the stable ingestion entry point.  ``raw_text`` is accepted
    separately so callers can supply content from disk, Git, an editor buffer,
    or another source without forcing filesystem I/O in this module.
    """

    return _default_chunker.chunk_file(file_path, raw_text)


__all__ = ["Chunk", "ContextAwareChunker", "chunk_file"]
