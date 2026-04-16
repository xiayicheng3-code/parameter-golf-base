from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import re
from typing import Any

import torch
from torch import Tensor, nn


TOKEN_OTHER = 0
TOKEN_ALPHA = 1
TOKEN_DIGIT = 2
TOKEN_SPACE = 3
TOKEN_NEWLINE = 4
TOKEN_PUNCT = 5
TOKEN_SLASH = 6
TOKEN_QUOTE = 7
TOKEN_ANGLE = 8
TOKEN_BRACKET = 9
TOKEN_OPERATOR = 10
TOKEN_BYTE = 11
TOKEN_CLASS_COUNT = 12

ROLE_OTHER = 0
ROLE_DETERMINER = 1
ROLE_PRONOUN = 2
ROLE_PREPOSITION = 3
ROLE_AUXILIARY = 4
ROLE_MODAL = 5
ROLE_CONJUNCTION = 6
ROLE_VERBISH = 7
ROLE_ADVERBISH = 8
ROLE_NOUNISH = 9
ROLE_HTML_TAG = 10
ROLE_HTML_ATTR = 11
ROLE_JSON_KEYWORD = 12
ROLE_URLISH = 13
ROLE_MARKDOWN = 14
ROLE_CODE = 15
ROLE_COUNT = 16

SHAPE_OTHER = 0
SHAPE_LOWER = 1
SHAPE_UPPER = 2
SHAPE_TITLE = 3
SHAPE_MIXED_ALPHA = 4
SHAPE_DIGIT = 5
SHAPE_ALNUM = 6
SHAPE_SYMBOL = 7
SHAPE_COUNT = 8

LINE_MID = 0
LINE_START = 1
LINE_AFTER_INDENT = 2
LINE_COUNT = 3

MODE_PLAIN = 0
MODE_HTML = 1
MODE_URL = 2
MODE_JSON = 3
MODE_MARKDOWN = 4
MODE_CODE = 5
MODE_COUNT = 6

QUOTE_NONE = 0
QUOTE_SINGLE = 1
QUOTE_DOUBLE = 2
QUOTE_BACKTICK = 3
QUOTE_COUNT = 4

SCOPE_NONE = 0
SCOPE_JSON_STRING = 1
SCOPE_HTML_ATTR = 2
SCOPE_CODE_STRING = 3
SCOPE_URL = 4
SCOPE_MARKDOWN_CODE = 5
SCOPE_COUNT = 6

PROSE_NEUTRAL = 0
PROSE_SENTENCE_START = 1
PROSE_AFTER_DETERMINER = 2
PROSE_AFTER_PREPOSITION = 3
PROSE_AFTER_AUX = 4
PROSE_AFTER_SUBJECT = 5
PROSE_AFTER_VERB = 6
PROSE_AFTER_PUNCT = 7
PROSE_COUNT = 8

DEPTH_COUNT = 5
LEN_BUCKET_COUNT = 8
RUN_BUCKET_COUNT = 6
BOOL_COUNT = 2

HTML_OUT = 0
HTML_AFTER_LT = 1
HTML_TAG_NAME = 2
HTML_ATTR_NAME = 3
HTML_AFTER_EQUALS = 4
HTML_QUOTED_ATTR = 5
HTML_CLOSING = 6
HTML_COUNT = 7

JSON_OUT = 0
JSON_EXPECT_KEY = 1
JSON_AFTER_KEY = 2
JSON_EXPECT_VALUE = 3
JSON_AFTER_VALUE = 4
JSON_IN_STRING = 5
JSON_COUNT = 6

URL_OUT = 0
URL_HOST = 1
URL_PATH = 2
URL_QUERY = 3
URL_FRAGMENT = 4
URL_COUNT = 5

MARKDOWN_OUT = 0
MARKDOWN_HEADING = 1
MARKDOWN_LIST = 2
MARKDOWN_QUOTE = 3
MARKDOWN_CODE_FENCE = 4
MARKDOWN_TABLE = 5
MARKDOWN_COUNT = 6

CODE_OUT = 0
CODE_EXPR = 1
CODE_STRING = 2
CODE_COMMENT = 3
CODE_BLOCK = 4
CODE_COUNT = 5

PUNCT_NONE = 0
PUNCT_SENTENCE = 1
PUNCT_COMMA = 2
PUNCT_COLON = 3
PUNCT_SEMICOLON = 4
PUNCT_OPEN = 5
PUNCT_COUNT = 6

STACK_NONE = 0
STACK_PAREN = 1
STACK_BRACKET = 2
STACK_BRACE = 3
STACK_QUOTE = 4
STACK_HTML = 5
STACK_JSON_OBJECT = 6
STACK_JSON_ARRAY = 7
STACK_COUNT = 8

TAG_OTHER = 0
TAG_LINK = 1
TAG_BLOCK = 2
TAG_INLINE = 3
TAG_LIST = 4
TAG_TABLE = 5
TAG_MEDIA = 6
TAG_SCRIPT_STYLE = 7
TAG_META = 8
TAG_COUNT = 9

JSON_CONTAINER_NONE = 0
JSON_CONTAINER_OBJECT = 1
JSON_CONTAINER_ARRAY = 2
JSON_CONTAINER_COUNT = 3

HEADING_LEVEL_COUNT = 7

INDENT_DELTA_NONE = 0
INDENT_DELTA_SAME = 1
INDENT_DELTA_INDENT = 2
INDENT_DELTA_DEDENT = 3
INDENT_DELTA_COUNT = 4

KEYVAL_NONE = 0
KEYVAL_KEY = 1
KEYVAL_VALUE = 2
KEYVAL_COUNT = 3

PROSE_CLAUSE_NONE = 0
PROSE_CLAUSE_NP = 1
PROSE_CLAUSE_VP = 2
PROSE_CLAUSE_PP = 3
PROSE_CLAUSE_SUBORD = 4
PROSE_CLAUSE_REL = 5
PROSE_CLAUSE_COUNT = 6

ABBREV_NONE = 0
ABBREV_PENDING_UPPER = 1
ABBREV_PENDING_LOWER = 2
ABBREV_CHAIN_UPPER = 3
ABBREV_CHAIN_LOWER = 4
ABBREV_COUNT = 5

EDGE_NONE = 0
EDGE_AFTER_OPEN = 1
EDGE_AFTER_CLOSE = 2
EDGE_AFTER_COLON = 3
EDGE_AFTER_EQUALS = 4
EDGE_AFTER_DOT = 5
EDGE_AFTER_SCHEME = 6
EDGE_AFTER_COMMA = 7
EDGE_AFTER_OPEN_QUOTE = 8
EDGE_AFTER_CLOSE_QUOTE = 9
EDGE_COUNT = 10

CHAIN_NONE = 0
CHAIN_AFTER_DOT = 1
CHAIN_AFTER_CALL_OPEN = 2
CHAIN_AFTER_INDEX_OPEN = 3
CHAIN_AFTER_CLOSE = 4
CHAIN_COUNT = 5

FEATURE_NAMES = (
    "indent_bucket",
    "mode",
    "html_phase",
    "json_depth",
    "json_expect",
    "url_phase",
    "markdown_phase",
    "code_phase",
    "quote_mode",
    "scope_state",
    "quote_len_bucket",
    "prose_state",
    "sentence_len_bucket",
    "clause_len_bucket",
    "bracket_depth",
    "same_class_run_bucket",
    "followup_edge_state",
    "stack_top_type",
    "stack_depth_bucket",
    "stack_parent_type",
    "html_top_tag_class",
    "html_tag_depth_bucket",
    "json_container_top",
    "markdown_heading_level",
    "markdown_list_depth",
    "markdown_table_col_bucket",
    "markdown_fence_state",
    "indent_delta",
    "blank_line_run_bucket",
    "member_chain_state",
    "key_value_side",
    "code_call_arg_bucket",
    "prose_clause_stack",
    "serial_edge_state",
    "abbrev_state",
)
FEATURE_SIZES = (
    LEN_BUCKET_COUNT,
    MODE_COUNT,
    HTML_COUNT,
    DEPTH_COUNT,
    JSON_COUNT,
    URL_COUNT,
    MARKDOWN_COUNT,
    CODE_COUNT,
    QUOTE_COUNT,
    SCOPE_COUNT,
    LEN_BUCKET_COUNT,
    PROSE_COUNT,
    LEN_BUCKET_COUNT,
    LEN_BUCKET_COUNT,
    DEPTH_COUNT,
    RUN_BUCKET_COUNT,
    EDGE_COUNT,
    STACK_COUNT,
    DEPTH_COUNT,
    STACK_COUNT,
    TAG_COUNT,
    DEPTH_COUNT,
    JSON_CONTAINER_COUNT,
    HEADING_LEVEL_COUNT,
    DEPTH_COUNT,
    LEN_BUCKET_COUNT,
    BOOL_COUNT,
    INDENT_DELTA_COUNT,
    RUN_BUCKET_COUNT,
    CHAIN_COUNT,
    KEYVAL_COUNT,
    LEN_BUCKET_COUNT,
    PROSE_CLAUSE_COUNT,
    EDGE_COUNT,
    ABBREV_COUNT,
)

F_SPACE = 1 << 0
F_NEWLINE = 1 << 1
F_SENT_END = 1 << 2
F_COMMA = 1 << 3
F_HTML_START = 1 << 4
F_HTML_END = 1 << 5
F_URLISH = 1 << 6
F_JSON_OPEN = 1 << 7
F_JSON_CLOSE = 1 << 8
F_PAREN_OPEN = 1 << 9
F_PAREN_CLOSE = 1 << 10
F_MARKDOWN = 1 << 11
F_CODEISH = 1 << 12
F_COLON = 1 << 13
F_SEMICOLON = 1 << 14
F_EQUALS = 1 << 15
F_HASH = 1 << 16
F_DASH = 1 << 17
F_PIPE = 1 << 18
F_QUERY = 1 << 19
F_FRAGMENT = 1 << 20
F_LBRACE = 1 << 21
F_RBRACE = 1 << 22
F_LBRACKET = 1 << 23
F_RBRACKET = 1 << 24
F_LPAREN = 1 << 25
F_RPAREN = 1 << 26
F_DOT = 1 << 27
F_SLASH = 1 << 28
F_BOS = 1 << 29
F_SCHEME_DELIM = 1 << 30
F_SINGLE_UPPER = 1 << 31
F_SINGLE_LOWER = 1 << 32

DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "their", "his", "her", "its"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "who", "which", "that"}
PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "from", "to", "of", "about", "into", "over", "under", "between", "through"}
AUXILIARIES = {"am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
CONJUNCTIONS = {"and", "or", "but", "if", "because", "while", "although", "since", "unless", "when", "where"}
HTML_TAGS = {"a", "p", "div", "span", "ul", "ol", "li", "table", "tr", "td", "th", "img", "script", "style", "meta", "link", "body", "html", "head"}
HTML_ATTRS = {"class", "id", "href", "src", "alt", "title", "style", "type", "name", "content", "rel", "role", "aria", "data"}
JSON_KEYWORDS = {"true", "false", "null"}
CODE_WORDS = {"def", "class", "return", "import", "from", "function", "const", "let", "var", "public", "private", "void", "int", "str", "self"}
URL_MARKERS = ("http", "www", "://", ".com", ".org", ".net", ".io", ".html", ".php")


@dataclass(frozen=True)
class GrammarVocabTables:
    token_class: Tensor
    lexical_role: Tensor
    shape: Tensor
    quote_kind: Tensor
    tag_class: Tensor
    flags: Tensor

    @property
    def vocab_size(self) -> int:
        return int(self.token_class.numel())


def _piece_text(sp: Any, token_id: int) -> str:
    piece = sp.id_to_piece(token_id)
    if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
        return ""
    if sp.is_byte(token_id):
        match = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece)
        if match is None:
            return piece
        return bytes([int(match.group(1), 16)]).decode("latin1")
    return piece.replace("▁", " ")


def _stripped_word(text: str) -> str:
    return text.strip().strip("\"'`“”‘’()[]{}<>.,:;!?").lower()


def _token_class(text: str, is_byte: bool) -> int:
    if is_byte:
        return TOKEN_BYTE
    if "\n" in text or "\r" in text:
        return TOKEN_NEWLINE
    if text and text.isspace():
        return TOKEN_SPACE
    stripped = text.strip()
    if not stripped:
        return TOKEN_SPACE if " " in text else TOKEN_OTHER
    if stripped.isdigit():
        return TOKEN_DIGIT
    if stripped.isalpha():
        return TOKEN_ALPHA
    if any(ch in stripped for ch in "\"'`“”‘’"):
        return TOKEN_QUOTE
    if any(ch in stripped for ch in "<>"):
        return TOKEN_ANGLE
    if any(ch in stripped for ch in "()[]{}"):
        return TOKEN_BRACKET
    if "/" in stripped or "\\" in stripped:
        return TOKEN_SLASH
    if any(ch in stripped for ch in "=+-*%&|^~"):
        return TOKEN_OPERATOR
    if any(ch in stripped for ch in ".,:;!?"):
        return TOKEN_PUNCT
    return TOKEN_OTHER


def _shape(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return SHAPE_OTHER
    if stripped.isdigit():
        return SHAPE_DIGIT
    if stripped.isalpha():
        if stripped.islower():
            return SHAPE_LOWER
        if stripped.isupper():
            return SHAPE_UPPER
        if stripped.istitle():
            return SHAPE_TITLE
        return SHAPE_MIXED_ALPHA
    if stripped.isalnum():
        return SHAPE_ALNUM
    return SHAPE_SYMBOL


def _lexical_role(text: str) -> int:
    word = _stripped_word(text)
    lowered = text.lower()
    if not word:
        if any(ch in text for ch in "#>*-[]()`"):
            return ROLE_MARKDOWN
        return ROLE_OTHER
    if any(marker in lowered for marker in URL_MARKERS):
        return ROLE_URLISH
    if word in DETERMINERS:
        return ROLE_DETERMINER
    if word in PRONOUNS:
        return ROLE_PRONOUN
    if word in PREPOSITIONS:
        return ROLE_PREPOSITION
    if word in AUXILIARIES:
        return ROLE_AUXILIARY
    if word in MODALS:
        return ROLE_MODAL
    if word in CONJUNCTIONS:
        return ROLE_CONJUNCTION
    if word in HTML_TAGS:
        return ROLE_HTML_TAG
    if word in HTML_ATTRS or word.startswith("aria") or word.startswith("data"):
        return ROLE_HTML_ATTR
    if word in JSON_KEYWORDS:
        return ROLE_JSON_KEYWORD
    if word in CODE_WORDS:
        return ROLE_CODE
    if word.endswith(("ing", "ed", "ify", "ise", "ize")):
        return ROLE_VERBISH
    if word.endswith("ly"):
        return ROLE_ADVERBISH
    if word.endswith(("tion", "ment", "ness", "ity", "ship", "ism", "er", "or")):
        return ROLE_NOUNISH
    return ROLE_OTHER


def _html_tag_class(text: str) -> int:
    word = _stripped_word(text)
    if word in {"a", "link"}:
        return TAG_LINK
    if word in {"div", "p", "section", "article", "main", "body", "html", "head", "header", "footer", "nav"}:
        return TAG_BLOCK
    if word in {"span", "b", "i", "em", "strong", "small", "label"}:
        return TAG_INLINE
    if word in {"ul", "ol", "li", "dl", "dt", "dd"}:
        return TAG_LIST
    if word in {"table", "tr", "td", "th", "thead", "tbody"}:
        return TAG_TABLE
    if word in {"img", "video", "audio", "source", "picture", "svg"}:
        return TAG_MEDIA
    if word in {"script", "style"}:
        return TAG_SCRIPT_STYLE
    if word in {"meta", "title", "base"}:
        return TAG_META
    return TAG_OTHER


def _quote_kind(text: str) -> int:
    if "`" in text:
        return QUOTE_BACKTICK
    if '"' in text or "“" in text or "”" in text:
        return QUOTE_DOUBLE
    if "'" in text or "‘" in text or "’" in text:
        return QUOTE_SINGLE
    return QUOTE_NONE


def _flags(text: str, role: int, is_bos: bool = False) -> int:
    flags = 0
    stripped = text.strip()
    if text.isspace() or " " in text or "\t" in text:
        flags |= F_SPACE
    if "\n" in text or "\r" in text:
        flags |= F_NEWLINE
    if any(ch in text for ch in ".!?"):
        flags |= F_SENT_END
    if "," in text:
        flags |= F_COMMA
    if ":" in text:
        flags |= F_COLON
    if ";" in text:
        flags |= F_SEMICOLON
    if "=" in text:
        flags |= F_EQUALS
    if "#" in text:
        flags |= F_HASH
    if "-" in text or "*" in text:
        flags |= F_DASH
    if "|" in text:
        flags |= F_PIPE
    if "?" in text:
        flags |= F_QUERY
    if "#" in text:
        flags |= F_FRAGMENT
    if "{" in text:
        flags |= F_LBRACE
    if "}" in text:
        flags |= F_RBRACE
    if "[" in text:
        flags |= F_LBRACKET
    if "]" in text:
        flags |= F_RBRACKET
    if "(" in text:
        flags |= F_LPAREN
    if ")" in text:
        flags |= F_RPAREN
    if "." in text:
        flags |= F_DOT
    if "/" in text or "\\" in text:
        flags |= F_SLASH
    if "://" in text:
        flags |= F_SCHEME_DELIM
    if "<" in text:
        flags |= F_HTML_START
    if ">" in text:
        flags |= F_HTML_END
    if role == ROLE_URLISH or any(marker in text.lower() for marker in URL_MARKERS):
        flags |= F_URLISH
    if any(ch in text for ch in "[{"):
        flags |= F_JSON_OPEN
    if any(ch in text for ch in "]}"):
        flags |= F_JSON_CLOSE
    if "(" in text:
        flags |= F_PAREN_OPEN
    if ")" in text:
        flags |= F_PAREN_CLOSE
    if role == ROLE_MARKDOWN or any(ch in text for ch in "#>*[]`"):
        flags |= F_MARKDOWN
    if role == ROLE_CODE or any(ch in text for ch in "=;{}"):
        flags |= F_CODEISH
    if is_bos:
        flags |= F_BOS
    if stripped.isalpha() and len(stripped) == 1:
        if stripped.isupper():
            flags |= F_SINGLE_UPPER
        elif stripped.islower():
            flags |= F_SINGLE_LOWER
    return flags


def build_vocab_tables_from_sentencepiece(sp: Any, vocab_size: int) -> GrammarVocabTables:
    table_size = max(int(vocab_size), int(sp.vocab_size()))
    token_class = torch.zeros(table_size, dtype=torch.long)
    lexical_role = torch.zeros(table_size, dtype=torch.long)
    shape = torch.zeros(table_size, dtype=torch.long)
    quote_kind = torch.zeros(table_size, dtype=torch.long)
    tag_class = torch.zeros(table_size, dtype=torch.long)
    flags = torch.zeros(table_size, dtype=torch.long)
    for token_id in range(int(sp.vocab_size())):
        text = _piece_text(sp, token_id)
        is_byte = bool(sp.is_byte(token_id))
        role = _lexical_role(text)
        token_class[token_id] = _token_class(text, is_byte)
        lexical_role[token_id] = role
        shape[token_id] = _shape(text)
        quote_kind[token_id] = _quote_kind(text)
        tag_class[token_id] = _html_tag_class(text)
        flags[token_id] = _flags(text, role, is_bos=(token_id == int(sp.bos_id())))
    return GrammarVocabTables(token_class, lexical_role, shape, quote_kind, tag_class, flags)


def _len_bucket(values: Tensor) -> Tensor:
    out = torch.zeros_like(values)
    out = torch.where(values >= 1, torch.ones_like(out), out)
    out = torch.where(values >= 2, torch.full_like(out, 2), out)
    out = torch.where(values >= 3, torch.full_like(out, 3), out)
    out = torch.where(values >= 5, torch.full_like(out, 4), out)
    out = torch.where(values >= 9, torch.full_like(out, 5), out)
    out = torch.where(values >= 15, torch.full_like(out, 6), out)
    out = torch.where(values >= 31, torch.full_like(out, 7), out)
    return out


def _run_bucket(values: Tensor) -> Tensor:
    out = torch.zeros_like(values)
    out = torch.where(values >= 1, torch.ones_like(out), out)
    out = torch.where(values >= 2, torch.full_like(out, 2), out)
    out = torch.where(values >= 3, torch.full_like(out, 3), out)
    out = torch.where(values >= 5, torch.full_like(out, 4), out)
    out = torch.where(values >= 9, torch.full_like(out, 5), out)
    return out


def _len_bucket_scalar(value: int) -> int:
    if value >= 31:
        return 7
    if value >= 15:
        return 6
    if value >= 9:
        return 5
    if value >= 5:
        return 4
    if value >= 3:
        return 3
    if value >= 2:
        return 2
    if value >= 1:
        return 1
    return 0


def _run_bucket_scalar(value: int) -> int:
    if value >= 9:
        return 5
    if value >= 5:
        return 4
    if value >= 3:
        return 3
    if value >= 2:
        return 2
    if value >= 1:
        return 1
    return 0


class GrammarStateEmbedding(nn.Module):
    def __init__(self, tables: GrammarVocabTables, dim: int, init_scale: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("token_class_lut", tables.token_class.long(), persistent=True)
        self.register_buffer("lexical_role_lut", tables.lexical_role.long(), persistent=True)
        self.register_buffer("shape_lut", tables.shape.long(), persistent=True)
        self.register_buffer("quote_kind_lut", tables.quote_kind.long(), persistent=True)
        self.register_buffer("tag_class_lut", tables.tag_class.long(), persistent=True)
        self.register_buffer("flags_lut", tables.flags.long(), persistent=True)
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim) for size in FEATURE_SIZES])
        self.feature_scale = nn.Parameter(torch.ones(len(FEATURE_SIZES), dtype=torch.float32))
        self.gate = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self._numpy_lut_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=dim ** -0.5)

    def _numpy_luts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._numpy_lut_cache is None:
            self._numpy_lut_cache = (
                self.token_class_lut.detach().cpu().numpy().astype(np.int16, copy=False),
                self.lexical_role_lut.detach().cpu().numpy().astype(np.int16, copy=False),
                self.shape_lut.detach().cpu().numpy().astype(np.int16, copy=False),
                self.quote_kind_lut.detach().cpu().numpy().astype(np.int16, copy=False),
                self.tag_class_lut.detach().cpu().numpy().astype(np.int16, copy=False),
                self.flags_lut.detach().cpu().numpy().astype(np.int64, copy=False),
            )
        return self._numpy_lut_cache

    def build_feature_ids_numpy_1d(self, input_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        steps = int(ids.shape[0])
        features = np.empty((steps, len(FEATURE_SIZES)), dtype=np.uint8)
        token_class_lut, lexical_role_lut, shape_lut, quote_kind_lut, tag_class_lut, flags_lut = self._numpy_luts()

        line_state = LINE_START
        indent_len = 0
        mode = MODE_PLAIN
        html_phase = HTML_OUT
        json_depth = 0
        json_expect = JSON_OUT
        url_phase = URL_OUT
        markdown_phase = MARKDOWN_OUT
        code_phase = CODE_OUT
        quote_mode = QUOTE_NONE
        scope_state = SCOPE_NONE
        quote_len = 0
        prose_state = PROSE_SENTENCE_START
        sentence_len = 0
        clause_len = 0
        bracket_depth = 0
        same_class_run = 0
        prev_class = TOKEN_OTHER
        followup_edge_state = EDGE_NONE
        stack_top_type = STACK_NONE
        stack_depth = 0
        stack_parent_type = STACK_NONE
        stack_slots = [STACK_NONE for _ in range(DEPTH_COUNT - 1)]
        html_top_tag_class = TAG_OTHER
        html_tag_depth = 0
        json_container_top = JSON_CONTAINER_NONE
        markdown_heading_level = 0
        markdown_list_depth = 0
        markdown_table_col = 0
        markdown_fence_state = 0
        last_indent_bucket = 0
        indent_delta = INDENT_DELTA_NONE
        blank_line_run = 0
        line_token_count = 0
        member_chain_state = CHAIN_NONE
        key_value_side = KEYVAL_NONE
        code_call_arg_count = 0
        prose_clause_stack = PROSE_CLAUSE_NONE
        serial_edge_state = EDGE_NONE
        abbrev_state = ABBREV_NONE

        for pos, tok in enumerate(ids.tolist()):
            tok_class = int(token_class_lut[tok])
            role = int(lexical_role_lut[tok])
            shape = int(shape_lut[tok])
            quote_kind = int(quote_kind_lut[tok])
            tag_class = int(tag_class_lut[tok])
            flags = int(flags_lut[tok])

            is_bos = bool(flags & F_BOS)
            is_space = bool(flags & F_SPACE)
            is_newline = bool(flags & F_NEWLINE)
            is_sent_end = bool(flags & F_SENT_END)
            is_comma = bool(flags & F_COMMA)
            starts_html = bool(flags & F_HTML_START)
            ends_html = bool(flags & F_HTML_END)
            is_urlish = bool(flags & F_URLISH)
            opens_json = bool(flags & F_JSON_OPEN)
            closes_json = bool(flags & F_JSON_CLOSE)
            opens_paren = bool(flags & F_PAREN_OPEN)
            closes_paren = bool(flags & F_PAREN_CLOSE)
            is_markdown = bool(flags & F_MARKDOWN)
            is_codeish = bool(flags & F_CODEISH)
            is_colon = bool(flags & F_COLON)
            is_semicolon = bool(flags & F_SEMICOLON)
            is_equals = bool(flags & F_EQUALS)
            is_hash = bool(flags & F_HASH)
            is_dash = bool(flags & F_DASH)
            is_pipe = bool(flags & F_PIPE)
            is_query = bool(flags & F_QUERY)
            is_fragment = bool(flags & F_FRAGMENT)
            opens_brace = bool(flags & F_LBRACE)
            closes_brace = bool(flags & F_RBRACE)
            opens_bracket = bool(flags & F_LBRACKET)
            closes_bracket = bool(flags & F_RBRACKET)
            opens_round = bool(flags & F_LPAREN)
            closes_round = bool(flags & F_RPAREN)
            is_dot = bool(flags & F_DOT)
            is_slash = bool(flags & F_SLASH)
            is_scheme_delim = bool(flags & F_SCHEME_DELIM)
            is_single_upper = bool(flags & F_SINGLE_UPPER)
            is_single_lower = bool(flags & F_SINGLE_LOWER)
            is_word = tok_class in (TOKEN_ALPHA, TOKEN_DIGIT)
            is_tokenish = not (is_space or is_newline)
            is_layout_token = tok_class in (TOKEN_SPACE, TOKEN_NEWLINE)
            line_like_start = line_state in (LINE_START, LINE_AFTER_INDENT)
            prev_html_phase = html_phase
            current_followup_edge = EDGE_NONE if is_layout_token else followup_edge_state
            current_member_chain = CHAIN_NONE if is_layout_token else member_chain_state

            same_class_run = same_class_run + 1 if tok_class == prev_class else 1
            prev_class = tok_class

            json_depth = max(0, min(DEPTH_COUNT - 1, json_depth + int(opens_json) - int(closes_json)))
            bracket_depth = max(0, min(DEPTH_COUNT - 1, bracket_depth + int(opens_paren) - int(closes_paren)))

            entering_quote = quote_kind != QUOTE_NONE and quote_mode == QUOTE_NONE
            leaving_quote = quote_kind != QUOTE_NONE and quote_mode == quote_kind
            if leaving_quote:
                quote_mode = QUOTE_NONE
            if entering_quote:
                quote_mode = quote_kind
            quote_len = 0 if quote_mode == QUOTE_NONE else quote_len + int(is_tokenish)

            any_struct_close = closes_brace or closes_bracket or closes_round or ends_html
            jsonish_context = (mode == MODE_JSON) or (json_depth > 0) or opens_json
            bracket_top = STACK_JSON_ARRAY if jsonish_context else STACK_BRACKET
            brace_top = STACK_JSON_OBJECT if jsonish_context else STACK_BRACE
            open_type = STACK_NONE
            if opens_round:
                open_type = STACK_PAREN
            if opens_bracket:
                open_type = bracket_top
            if opens_brace:
                open_type = brace_top
            if starts_html:
                open_type = STACK_HTML
            if open_type != STACK_NONE:
                open_slot = min(stack_depth, DEPTH_COUNT - 2)
                stack_slots[open_slot] = open_type
            stack_depth = max(0, min(DEPTH_COUNT - 1, stack_depth + int(open_type != STACK_NONE) - int(any_struct_close)))
            stack_top_type = stack_slots[stack_depth - 1] if stack_depth > 0 else STACK_NONE
            stack_parent_type = stack_slots[stack_depth - 2] if stack_depth > 1 else STACK_NONE

            protected_scope = ((json_depth > 0) and (quote_mode != QUOTE_NONE)) or (html_phase == HTML_QUOTED_ATTR)
            if mode == MODE_URL and (is_space or is_newline or ends_html):
                mode = MODE_PLAIN
            if starts_html:
                mode = MODE_HTML
            if is_urlish and (not is_space) and (not is_newline) and (not protected_scope) and mode in (MODE_PLAIN, MODE_URL):
                mode = MODE_URL
            if json_depth > 0 and mode == MODE_PLAIN:
                mode = MODE_JSON
            if json_depth == 0 and mode == MODE_JSON:
                mode = MODE_PLAIN
            if is_markdown and line_like_start and mode == MODE_PLAIN:
                mode = MODE_MARKDOWN
            if is_codeish and mode == MODE_PLAIN and not protected_scope:
                mode = MODE_CODE
            if is_newline and mode in (MODE_MARKDOWN, MODE_CODE):
                mode = MODE_PLAIN
            if mode == MODE_HTML and ends_html:
                mode = MODE_PLAIN

            if starts_html:
                html_phase = HTML_AFTER_LT
            if html_phase == HTML_AFTER_LT and is_slash:
                html_phase = HTML_CLOSING
            if html_phase == HTML_AFTER_LT and (tok_class == TOKEN_ALPHA or role == ROLE_HTML_TAG):
                html_phase = HTML_TAG_NAME
            if html_phase == HTML_TAG_NAME and is_space:
                html_phase = HTML_ATTR_NAME
            if html_phase == HTML_ATTR_NAME and is_equals:
                html_phase = HTML_AFTER_EQUALS
            if html_phase == HTML_AFTER_EQUALS and quote_kind != QUOTE_NONE:
                html_phase = HTML_QUOTED_ATTR
            if html_phase == HTML_QUOTED_ATTR and leaving_quote:
                html_phase = HTML_ATTR_NAME
            opening_tag_name = prev_html_phase == HTML_AFTER_LT and tag_class != TAG_OTHER
            closing_tag_name = prev_html_phase == HTML_CLOSING and tag_class != TAG_OTHER
            html_tag_depth = max(0, min(DEPTH_COUNT - 1, html_tag_depth + int(opening_tag_name) - int(closing_tag_name)))
            if opening_tag_name:
                html_top_tag_class = tag_class
            if html_tag_depth == 0:
                html_top_tag_class = TAG_OTHER
            if ends_html or is_newline:
                html_phase = HTML_OUT

            if opens_brace:
                json_container_top = JSON_CONTAINER_OBJECT
            if opens_bracket:
                json_container_top = JSON_CONTAINER_ARRAY
            if json_depth == 0:
                json_container_top = JSON_CONTAINER_NONE
            json_expect = JSON_OUT if json_depth == 0 else json_expect
            if opens_json:
                json_expect = JSON_EXPECT_KEY
            if json_depth > 0 and is_comma:
                json_expect = JSON_EXPECT_KEY
            if json_depth > 0 and is_colon:
                json_expect = JSON_EXPECT_VALUE
            if json_depth > 0 and quote_mode != QUOTE_NONE:
                json_expect = JSON_IN_STRING
            if json_depth > 0 and leaving_quote and json_expect == JSON_IN_STRING:
                json_expect = JSON_AFTER_VALUE
            if json_depth > 0 and role == ROLE_JSON_KEYWORD:
                json_expect = JSON_AFTER_VALUE
            if closes_json and json_depth == 0:
                json_expect = JSON_OUT

            if is_urlish and (not is_space) and (not is_newline):
                url_phase = URL_HOST
            if url_phase != URL_OUT and tok_class == TOKEN_SLASH and not is_scheme_delim:
                url_phase = URL_PATH
            if url_phase != URL_OUT and is_query:
                url_phase = URL_QUERY
            if url_phase != URL_OUT and is_fragment:
                url_phase = URL_FRAGMENT
            if url_phase != URL_OUT and (is_space or is_newline or ends_html):
                url_phase = URL_OUT

            if is_newline:
                markdown_phase = MARKDOWN_OUT
                markdown_table_col = 0
                markdown_heading_level = 0
                markdown_list_depth = 0
            if line_like_start and is_hash:
                markdown_phase = MARKDOWN_HEADING
                markdown_heading_level = min(HEADING_LEVEL_COUNT - 1, markdown_heading_level + 1)
            if line_like_start and is_dash:
                markdown_phase = MARKDOWN_LIST
                markdown_list_depth = min(DEPTH_COUNT - 1, _len_bucket_scalar(indent_len))
            if line_like_start and starts_html:
                markdown_phase = MARKDOWN_QUOTE
            if is_pipe:
                markdown_phase = MARKDOWN_TABLE
                markdown_table_col += 1
            if quote_kind == QUOTE_BACKTICK:
                markdown_phase = MARKDOWN_CODE_FENCE
            if line_like_start and quote_kind == QUOTE_BACKTICK:
                markdown_fence_state = 1 - markdown_fence_state

            if is_newline and bracket_depth == 0:
                code_phase = CODE_OUT
            if is_codeish or mode == MODE_CODE:
                code_phase = CODE_EXPR
            if quote_mode != QUOTE_NONE and mode == MODE_CODE:
                code_phase = CODE_STRING
            if is_hash and mode == MODE_CODE:
                code_phase = CODE_COMMENT
            if bracket_depth > 0:
                code_phase = CODE_BLOCK
            if opens_round:
                code_call_arg_count = 0
            if bracket_depth > 0 and is_comma:
                code_call_arg_count += 1
            if closes_round:
                code_call_arg_count = 0

            prev_abbrev_state = abbrev_state
            dot_in_abbrev = is_dot and prev_abbrev_state in (
                ABBREV_PENDING_UPPER,
                ABBREV_PENDING_LOWER,
                ABBREV_CHAIN_UPPER,
                ABBREV_CHAIN_LOWER,
            )
            effective_sent_end = is_sent_end and not dot_in_abbrev
            if effective_sent_end:
                prose_state = PROSE_SENTENCE_START
            if is_comma:
                prose_state = PROSE_AFTER_PUNCT
            if role == ROLE_DETERMINER:
                prose_state = PROSE_AFTER_DETERMINER
            if role == ROLE_PREPOSITION:
                prose_state = PROSE_AFTER_PREPOSITION
            if role in (ROLE_AUXILIARY, ROLE_MODAL):
                prose_state = PROSE_AFTER_AUX
            subject_like = role in (ROLE_PRONOUN, ROLE_NOUNISH) or shape == SHAPE_TITLE
            if subject_like and mode == MODE_PLAIN:
                prose_state = PROSE_AFTER_SUBJECT
            if role == ROLE_VERBISH and mode == MODE_PLAIN:
                prose_state = PROSE_AFTER_VERB
            if mode not in (MODE_PLAIN, MODE_MARKDOWN):
                prose_state = PROSE_NEUTRAL
            if effective_sent_end or is_newline:
                prose_clause_stack = PROSE_CLAUSE_NONE
            if role == ROLE_CONJUNCTION:
                prose_clause_stack = PROSE_CLAUSE_SUBORD
            if role == ROLE_PREPOSITION:
                prose_clause_stack = PROSE_CLAUSE_PP
            if role in (ROLE_AUXILIARY, ROLE_MODAL, ROLE_VERBISH):
                prose_clause_stack = PROSE_CLAUSE_VP
            if subject_like or role == ROLE_DETERMINER:
                prose_clause_stack = PROSE_CLAUSE_NP
            if mode not in (MODE_PLAIN, MODE_MARKDOWN):
                prose_clause_stack = PROSE_CLAUSE_NONE

            if is_space or is_newline:
                abbrev_state = ABBREV_NONE
            if is_single_upper:
                abbrev_state = ABBREV_PENDING_UPPER
            if is_single_lower:
                abbrev_state = ABBREV_PENDING_LOWER
            if is_dot and prev_abbrev_state == ABBREV_PENDING_UPPER:
                abbrev_state = ABBREV_CHAIN_UPPER
            if is_dot and prev_abbrev_state == ABBREV_PENDING_LOWER:
                abbrev_state = ABBREV_CHAIN_LOWER
            if is_word and not (is_single_upper or is_single_lower):
                abbrev_state = ABBREV_NONE
            if effective_sent_end and not is_dot:
                abbrev_state = ABBREV_NONE

            sentence_len = 0 if (effective_sent_end or is_newline) else sentence_len + int(is_tokenish)
            clause_reset = effective_sent_end or is_newline or is_comma or is_colon or is_semicolon
            clause_len = 0 if clause_reset else clause_len + int(is_tokenish)

            serial_edge_state = EDGE_NONE
            if opens_round or opens_brace or opens_bracket:
                serial_edge_state = EDGE_AFTER_OPEN
            if closes_round or closes_brace or closes_bracket:
                serial_edge_state = EDGE_AFTER_CLOSE
            if is_comma:
                serial_edge_state = EDGE_AFTER_COMMA
            if is_colon:
                serial_edge_state = EDGE_AFTER_COLON
            if is_equals:
                serial_edge_state = EDGE_AFTER_EQUALS
            if is_scheme_delim:
                serial_edge_state = EDGE_AFTER_SCHEME
            if is_dot and not dot_in_abbrev:
                serial_edge_state = EDGE_AFTER_DOT
            if entering_quote:
                serial_edge_state = EDGE_AFTER_OPEN_QUOTE
            if leaving_quote:
                serial_edge_state = EDGE_AFTER_CLOSE_QUOTE
            if not is_layout_token:
                followup_edge_state = EDGE_NONE
            if serial_edge_state != EDGE_NONE:
                followup_edge_state = serial_edge_state

            active_code_context = (mode == MODE_CODE) or (
                code_phase != CODE_OUT and (not protected_scope) and mode not in (MODE_HTML, MODE_JSON, MODE_URL)
            )
            next_member_chain = CHAIN_NONE
            if mode not in (MODE_URL, MODE_HTML, MODE_JSON) and is_dot and (not dot_in_abbrev) and (not is_space) and (not is_newline):
                next_member_chain = CHAIN_AFTER_DOT
            if active_code_context and opens_round:
                next_member_chain = CHAIN_AFTER_CALL_OPEN
            if active_code_context and opens_bracket:
                next_member_chain = CHAIN_AFTER_INDEX_OPEN
            if active_code_context and (closes_round or closes_bracket):
                next_member_chain = CHAIN_AFTER_CLOSE
            member_chain_state = next_member_chain

            if is_bos or is_newline or is_semicolon:
                key_value_side = KEYVAL_NONE
            if mode == MODE_HTML and html_phase in (HTML_OUT, HTML_TAG_NAME, HTML_CLOSING):
                key_value_side = KEYVAL_NONE
            if html_phase == HTML_ATTR_NAME:
                key_value_side = KEYVAL_KEY
            if html_phase in (HTML_AFTER_EQUALS, HTML_QUOTED_ATTR):
                key_value_side = KEYVAL_VALUE
            if opens_json and opens_brace:
                key_value_side = KEYVAL_KEY
            if opens_json and opens_bracket:
                key_value_side = KEYVAL_VALUE
            if json_depth > 0 and json_container_top == JSON_CONTAINER_OBJECT and is_comma:
                key_value_side = KEYVAL_KEY
            if json_depth > 0 and json_container_top == JSON_CONTAINER_ARRAY and is_comma:
                key_value_side = KEYVAL_VALUE
            if json_depth > 0 and is_colon:
                key_value_side = KEYVAL_VALUE
            if active_code_context and key_value_side == KEYVAL_NONE and is_tokenish and not is_equals:
                key_value_side = KEYVAL_KEY
            if active_code_context and is_equals:
                key_value_side = KEYVAL_VALUE
            if json_depth == 0 and mode not in (MODE_HTML, MODE_CODE) and html_phase == HTML_OUT:
                key_value_side = KEYVAL_NONE
            plain_structured_context = mode in (MODE_PLAIN, MODE_MARKDOWN) and markdown_fence_state == 0 and quote_mode == QUOTE_NONE
            if plain_structured_context and current_followup_edge in (EDGE_AFTER_COLON, EDGE_AFTER_EQUALS):
                key_value_side = KEYVAL_VALUE

            blank_line = is_newline and line_token_count == 0
            if blank_line:
                blank_line_run += 1
            if is_newline and not blank_line:
                blank_line_run = 0
            if is_newline:
                line_token_count = 0
            elif is_tokenish and (not is_space) and (not is_newline):
                line_token_count += 1
            curr_indent_bucket = _len_bucket_scalar(indent_len)
            line_content_start = line_like_start and is_tokenish and (not is_space) and (not is_newline)
            if is_newline:
                indent_delta = INDENT_DELTA_NONE
            if line_content_start and curr_indent_bucket == last_indent_bucket:
                indent_delta = INDENT_DELTA_SAME
            if line_content_start and curr_indent_bucket > last_indent_bucket:
                indent_delta = INDENT_DELTA_INDENT
            if line_content_start and curr_indent_bucket < last_indent_bucket:
                indent_delta = INDENT_DELTA_DEDENT
            if line_content_start:
                last_indent_bucket = curr_indent_bucket
            if is_newline:
                indent_len = 0
            elif line_like_start and is_space:
                indent_len += 1
            if is_newline:
                line_state = LINE_START
            elif line_state == LINE_START and is_space:
                line_state = LINE_AFTER_INDENT
            elif is_word or ((not is_space) and (not is_newline)):
                line_state = LINE_MID

            scope_state = SCOPE_NONE
            if url_phase != URL_OUT:
                scope_state = SCOPE_URL
            if mode == MODE_CODE and quote_mode != QUOTE_NONE:
                scope_state = SCOPE_CODE_STRING
            if markdown_fence_state > 0:
                scope_state = SCOPE_MARKDOWN_CODE
            if html_phase == HTML_QUOTED_ATTR:
                scope_state = SCOPE_HTML_ATTR
            if json_depth > 0 and quote_mode != QUOTE_NONE:
                scope_state = SCOPE_JSON_STRING

            if is_bos:
                line_state = LINE_START
                indent_len = 0
                mode = MODE_PLAIN
                html_phase = HTML_OUT
                json_depth = 0
                json_expect = JSON_OUT
                url_phase = URL_OUT
                markdown_phase = MARKDOWN_OUT
                code_phase = CODE_OUT
                quote_mode = QUOTE_NONE
                scope_state = SCOPE_NONE
                quote_len = 0
                prose_state = PROSE_SENTENCE_START
                sentence_len = 0
                clause_len = 0
                bracket_depth = 0
                same_class_run = 0
                prev_class = TOKEN_OTHER
                followup_edge_state = EDGE_NONE
                stack_top_type = STACK_NONE
                stack_depth = 0
                stack_parent_type = STACK_NONE
                stack_slots = [STACK_NONE for _ in range(DEPTH_COUNT - 1)]
                html_top_tag_class = TAG_OTHER
                html_tag_depth = 0
                json_container_top = JSON_CONTAINER_NONE
                markdown_heading_level = 0
                markdown_list_depth = 0
                markdown_table_col = 0
                markdown_fence_state = 0
                last_indent_bucket = 0
                indent_delta = INDENT_DELTA_NONE
                blank_line_run = 0
                line_token_count = 0
                member_chain_state = CHAIN_NONE
                key_value_side = KEYVAL_NONE
                code_call_arg_count = 0
                prose_clause_stack = PROSE_CLAUSE_NONE
                serial_edge_state = EDGE_NONE
                abbrev_state = ABBREV_NONE
                current_followup_edge = EDGE_NONE
                current_member_chain = CHAIN_NONE

            features[pos, 0] = _len_bucket_scalar(indent_len)
            features[pos, 1] = mode
            features[pos, 2] = html_phase
            features[pos, 3] = json_depth
            features[pos, 4] = json_expect
            features[pos, 5] = url_phase
            features[pos, 6] = markdown_phase
            features[pos, 7] = code_phase
            features[pos, 8] = quote_mode
            features[pos, 9] = scope_state
            features[pos, 10] = _len_bucket_scalar(quote_len)
            features[pos, 11] = prose_state
            features[pos, 12] = _len_bucket_scalar(sentence_len)
            features[pos, 13] = _len_bucket_scalar(clause_len)
            features[pos, 14] = bracket_depth
            features[pos, 15] = _run_bucket_scalar(same_class_run)
            features[pos, 16] = current_followup_edge
            features[pos, 17] = stack_top_type
            features[pos, 18] = stack_depth
            features[pos, 19] = stack_parent_type
            features[pos, 20] = html_top_tag_class
            features[pos, 21] = html_tag_depth
            features[pos, 22] = json_container_top
            features[pos, 23] = markdown_heading_level
            features[pos, 24] = markdown_list_depth
            features[pos, 25] = _len_bucket_scalar(markdown_table_col)
            features[pos, 26] = markdown_fence_state
            features[pos, 27] = indent_delta
            features[pos, 28] = _run_bucket_scalar(blank_line_run)
            features[pos, 29] = current_member_chain
            features[pos, 30] = key_value_side
            features[pos, 31] = _len_bucket_scalar(code_call_arg_count)
            features[pos, 32] = prose_clause_stack
            features[pos, 33] = serial_edge_state
            features[pos, 34] = abbrev_state
        return features

    def build_feature_ids(self, input_ids: Tensor) -> Tensor:
        ids = input_ids.long()
        if ids.ndim != 2:
            raise ValueError(f"GrammarStateEmbedding expects input_ids with shape (batch, time), got {tuple(ids.shape)}")
        batch, steps = ids.shape
        device = ids.device
        features = torch.empty(batch, steps, len(FEATURE_SIZES), device=device, dtype=torch.long)
        line_state = torch.full((batch,), LINE_START, device=device, dtype=torch.long)
        indent_len = torch.zeros(batch, device=device, dtype=torch.long)
        mode = torch.zeros(batch, device=device, dtype=torch.long)
        html_phase = torch.zeros(batch, device=device, dtype=torch.long)
        json_depth = torch.zeros(batch, device=device, dtype=torch.long)
        json_expect = torch.zeros(batch, device=device, dtype=torch.long)
        url_phase = torch.zeros(batch, device=device, dtype=torch.long)
        markdown_phase = torch.zeros(batch, device=device, dtype=torch.long)
        code_phase = torch.zeros(batch, device=device, dtype=torch.long)
        quote_mode = torch.zeros(batch, device=device, dtype=torch.long)
        scope_state = torch.zeros(batch, device=device, dtype=torch.long)
        quote_len = torch.zeros(batch, device=device, dtype=torch.long)
        prose_state = torch.full((batch,), PROSE_SENTENCE_START, device=device, dtype=torch.long)
        sentence_len = torch.zeros(batch, device=device, dtype=torch.long)
        clause_len = torch.zeros(batch, device=device, dtype=torch.long)
        bracket_depth = torch.zeros(batch, device=device, dtype=torch.long)
        same_class_run = torch.zeros(batch, device=device, dtype=torch.long)
        prev_class = torch.full((batch,), TOKEN_OTHER, device=device, dtype=torch.long)
        followup_edge_state = torch.zeros(batch, device=device, dtype=torch.long)
        stack_top_type = torch.zeros(batch, device=device, dtype=torch.long)
        stack_depth = torch.zeros(batch, device=device, dtype=torch.long)
        stack_parent_type = torch.zeros(batch, device=device, dtype=torch.long)
        stack_slots = torch.zeros(batch, DEPTH_COUNT - 1, device=device, dtype=torch.long)
        html_top_tag_class = torch.zeros(batch, device=device, dtype=torch.long)
        html_tag_depth = torch.zeros(batch, device=device, dtype=torch.long)
        json_container_top = torch.zeros(batch, device=device, dtype=torch.long)
        markdown_heading_level = torch.zeros(batch, device=device, dtype=torch.long)
        markdown_list_depth = torch.zeros(batch, device=device, dtype=torch.long)
        markdown_table_col = torch.zeros(batch, device=device, dtype=torch.long)
        markdown_fence_state = torch.zeros(batch, device=device, dtype=torch.long)
        last_indent_bucket = torch.zeros(batch, device=device, dtype=torch.long)
        indent_delta = torch.zeros(batch, device=device, dtype=torch.long)
        blank_line_run = torch.zeros(batch, device=device, dtype=torch.long)
        line_token_count = torch.zeros(batch, device=device, dtype=torch.long)
        member_chain_state = torch.zeros(batch, device=device, dtype=torch.long)
        key_value_side = torch.zeros(batch, device=device, dtype=torch.long)
        code_call_arg_count = torch.zeros(batch, device=device, dtype=torch.long)
        prose_clause_stack = torch.zeros(batch, device=device, dtype=torch.long)
        serial_edge_state = torch.zeros(batch, device=device, dtype=torch.long)
        abbrev_state = torch.zeros(batch, device=device, dtype=torch.long)
        zero = torch.zeros(batch, device=device, dtype=torch.long)
        line_start_state = torch.full((batch,), LINE_START, device=device, dtype=torch.long)
        prose_start_state = torch.full((batch,), PROSE_SENTENCE_START, device=device, dtype=torch.long)
        other_class_state = torch.full((batch,), TOKEN_OTHER, device=device, dtype=torch.long)

        for pos in range(steps):
            tok = ids[:, pos]
            tok_class = self.token_class_lut[tok]
            role = self.lexical_role_lut[tok]
            shape = self.shape_lut[tok]
            quote_kind = self.quote_kind_lut[tok]
            tag_class = self.tag_class_lut[tok]
            flags = self.flags_lut[tok]

            is_bos = (flags & F_BOS) != 0
            is_space = (flags & F_SPACE) != 0
            is_newline = (flags & F_NEWLINE) != 0
            is_sent_end = (flags & F_SENT_END) != 0
            is_comma = (flags & F_COMMA) != 0
            starts_html = (flags & F_HTML_START) != 0
            ends_html = (flags & F_HTML_END) != 0
            is_urlish = (flags & F_URLISH) != 0
            opens_json = (flags & F_JSON_OPEN) != 0
            closes_json = (flags & F_JSON_CLOSE) != 0
            opens_paren = (flags & F_PAREN_OPEN) != 0
            closes_paren = (flags & F_PAREN_CLOSE) != 0
            is_markdown = (flags & F_MARKDOWN) != 0
            is_codeish = (flags & F_CODEISH) != 0
            is_colon = (flags & F_COLON) != 0
            is_semicolon = (flags & F_SEMICOLON) != 0
            is_equals = (flags & F_EQUALS) != 0
            is_hash = (flags & F_HASH) != 0
            is_dash = (flags & F_DASH) != 0
            is_pipe = (flags & F_PIPE) != 0
            is_query = (flags & F_QUERY) != 0
            is_fragment = (flags & F_FRAGMENT) != 0
            opens_brace = (flags & F_LBRACE) != 0
            closes_brace = (flags & F_RBRACE) != 0
            opens_bracket = (flags & F_LBRACKET) != 0
            closes_bracket = (flags & F_RBRACKET) != 0
            opens_round = (flags & F_LPAREN) != 0
            closes_round = (flags & F_RPAREN) != 0
            is_dot = (flags & F_DOT) != 0
            is_slash = (flags & F_SLASH) != 0
            is_scheme_delim = (flags & F_SCHEME_DELIM) != 0
            is_single_upper = (flags & F_SINGLE_UPPER) != 0
            is_single_lower = (flags & F_SINGLE_LOWER) != 0
            is_word = (tok_class == TOKEN_ALPHA) | (tok_class == TOKEN_DIGIT)
            is_tokenish = ~(is_space | is_newline)
            is_layout_token = (tok_class == TOKEN_SPACE) | (tok_class == TOKEN_NEWLINE)
            line_like_start = (line_state == LINE_START) | (line_state == LINE_AFTER_INDENT)
            prev_html_phase = html_phase
            current_followup_edge = torch.where(is_layout_token, zero, followup_edge_state)
            current_member_chain = torch.where(is_layout_token, zero, member_chain_state)

            same_class_run = torch.where(tok_class == prev_class, same_class_run + 1, torch.ones_like(same_class_run))
            prev_class = tok_class

            json_delta = opens_json.long() - closes_json.long()
            json_depth = torch.clamp(json_depth + json_delta, 0, DEPTH_COUNT - 1)
            paren_delta = opens_paren.long() - closes_paren.long()
            bracket_depth = torch.clamp(bracket_depth + paren_delta, 0, DEPTH_COUNT - 1)

            entering_quote = (quote_kind != QUOTE_NONE) & (quote_mode == QUOTE_NONE)
            leaving_quote = (quote_kind != QUOTE_NONE) & (quote_mode == quote_kind)
            quote_mode = torch.where(leaving_quote, zero, quote_mode)
            quote_mode = torch.where(entering_quote, quote_kind, quote_mode)
            quote_len = torch.where(quote_mode == QUOTE_NONE, zero, quote_len + is_tokenish.long())

            any_struct_open = opens_brace | opens_bracket | opens_round | starts_html
            any_struct_close = closes_brace | closes_bracket | closes_round | ends_html
            jsonish_context = (mode == MODE_JSON) | (json_depth > 0) | opens_json
            bracket_top = torch.where(
                jsonish_context,
                torch.full_like(stack_top_type, STACK_JSON_ARRAY),
                torch.full_like(stack_top_type, STACK_BRACKET),
            )
            brace_top = torch.where(
                jsonish_context,
                torch.full_like(stack_top_type, STACK_JSON_OBJECT),
                torch.full_like(stack_top_type, STACK_BRACE),
            )
            open_type = torch.where(opens_round, torch.full_like(stack_top_type, STACK_PAREN), zero)
            open_type = torch.where(opens_bracket, bracket_top, open_type)
            open_type = torch.where(opens_brace, brace_top, open_type)
            open_type = torch.where(starts_html, torch.full_like(stack_top_type, STACK_HTML), open_type)
            open_mask = open_type != STACK_NONE
            open_slot = torch.clamp(stack_depth, 0, DEPTH_COUNT - 2)
            for slot_idx in range(DEPTH_COUNT - 1):
                slot_mask = open_mask & (open_slot == slot_idx)
                stack_slots[:, slot_idx] = torch.where(slot_mask, open_type, stack_slots[:, slot_idx])
            stack_depth = torch.clamp(stack_depth + open_mask.long() - any_struct_close.long(), 0, DEPTH_COUNT - 1)
            top_slot = torch.clamp(stack_depth - 1, 0, DEPTH_COUNT - 2)
            parent_slot = torch.clamp(stack_depth - 2, 0, DEPTH_COUNT - 2)
            stack_top_type = zero.clone()
            stack_parent_type = zero.clone()
            for slot_idx in range(DEPTH_COUNT - 1):
                stack_top_type = torch.where((stack_depth > 0) & (top_slot == slot_idx), stack_slots[:, slot_idx], stack_top_type)
                stack_parent_type = torch.where((stack_depth > 1) & (parent_slot == slot_idx), stack_slots[:, slot_idx], stack_parent_type)

            protected_scope = ((json_depth > 0) & (quote_mode != QUOTE_NONE)) | (html_phase == HTML_QUOTED_ATTR)
            mode = torch.where((mode == MODE_URL) & (is_space | is_newline | ends_html), zero, mode)
            mode = torch.where(starts_html, torch.full_like(mode, MODE_HTML), mode)
            mode = torch.where(is_urlish & ~is_space & ~is_newline & ~protected_scope & ((mode == MODE_PLAIN) | (mode == MODE_URL)), torch.full_like(mode, MODE_URL), mode)
            mode = torch.where((json_depth > 0) & (mode == MODE_PLAIN), torch.full_like(mode, MODE_JSON), mode)
            mode = torch.where((json_depth == 0) & (mode == MODE_JSON), zero, mode)
            mode = torch.where(is_markdown & line_like_start & (mode == MODE_PLAIN), torch.full_like(mode, MODE_MARKDOWN), mode)
            mode = torch.where(is_codeish & (mode == MODE_PLAIN) & ~protected_scope, torch.full_like(mode, MODE_CODE), mode)
            mode = torch.where(is_newline & ((mode == MODE_MARKDOWN) | (mode == MODE_CODE)), zero, mode)
            mode = torch.where((mode == MODE_HTML) & ends_html, zero, mode)

            html_phase = torch.where(starts_html, torch.full_like(html_phase, HTML_AFTER_LT), html_phase)
            html_phase = torch.where(
                (html_phase == HTML_AFTER_LT) & is_slash,
                torch.full_like(html_phase, HTML_CLOSING),
                html_phase,
            )
            html_phase = torch.where(
                (html_phase == HTML_AFTER_LT) & ((tok_class == TOKEN_ALPHA) | (role == ROLE_HTML_TAG)),
                torch.full_like(html_phase, HTML_TAG_NAME),
                html_phase,
            )
            html_phase = torch.where((html_phase == HTML_TAG_NAME) & is_space, torch.full_like(html_phase, HTML_ATTR_NAME), html_phase)
            html_phase = torch.where((html_phase == HTML_ATTR_NAME) & is_equals, torch.full_like(html_phase, HTML_AFTER_EQUALS), html_phase)
            html_phase = torch.where(
                (html_phase == HTML_AFTER_EQUALS) & (quote_kind != QUOTE_NONE),
                torch.full_like(html_phase, HTML_QUOTED_ATTR),
                html_phase,
            )
            html_phase = torch.where((html_phase == HTML_QUOTED_ATTR) & leaving_quote, torch.full_like(html_phase, HTML_ATTR_NAME), html_phase)
            opening_tag_name = (prev_html_phase == HTML_AFTER_LT) & (tag_class != TAG_OTHER)
            closing_tag_name = (prev_html_phase == HTML_CLOSING) & (tag_class != TAG_OTHER)
            html_tag_depth = torch.clamp(html_tag_depth + opening_tag_name.long() - closing_tag_name.long(), 0, DEPTH_COUNT - 1)
            html_top_tag_class = torch.where(opening_tag_name, tag_class, html_top_tag_class)
            html_top_tag_class = torch.where(html_tag_depth == 0, zero, html_top_tag_class)
            html_phase = torch.where(ends_html | is_newline, zero, html_phase)

            json_container_top = torch.where(opens_brace, torch.full_like(json_container_top, JSON_CONTAINER_OBJECT), json_container_top)
            json_container_top = torch.where(opens_bracket, torch.full_like(json_container_top, JSON_CONTAINER_ARRAY), json_container_top)
            json_container_top = torch.where((json_depth == 0) | closes_json, torch.where(json_depth == 0, zero, json_container_top), json_container_top)
            json_expect = torch.where(json_depth == 0, zero, json_expect)
            json_expect = torch.where(opens_json, torch.full_like(json_expect, JSON_EXPECT_KEY), json_expect)
            json_expect = torch.where((json_depth > 0) & is_comma, torch.full_like(json_expect, JSON_EXPECT_KEY), json_expect)
            json_expect = torch.where((json_depth > 0) & is_colon, torch.full_like(json_expect, JSON_EXPECT_VALUE), json_expect)
            json_expect = torch.where(
                (json_depth > 0) & (quote_mode != QUOTE_NONE),
                torch.full_like(json_expect, JSON_IN_STRING),
                json_expect,
            )
            json_expect = torch.where(
                (json_depth > 0) & leaving_quote & (json_expect == JSON_IN_STRING),
                torch.full_like(json_expect, JSON_AFTER_VALUE),
                json_expect,
            )
            json_expect = torch.where((json_depth > 0) & (role == ROLE_JSON_KEYWORD), torch.full_like(json_expect, JSON_AFTER_VALUE), json_expect)
            json_expect = torch.where(closes_json & (json_depth == 0), zero, json_expect)

            url_phase = torch.where(is_urlish & ~is_space & ~is_newline, torch.full_like(url_phase, URL_HOST), url_phase)
            url_phase = torch.where((url_phase != URL_OUT) & (tok_class == TOKEN_SLASH) & ~is_scheme_delim, torch.full_like(url_phase, URL_PATH), url_phase)
            url_phase = torch.where((url_phase != URL_OUT) & is_query, torch.full_like(url_phase, URL_QUERY), url_phase)
            url_phase = torch.where((url_phase != URL_OUT) & is_fragment, torch.full_like(url_phase, URL_FRAGMENT), url_phase)
            url_phase = torch.where((url_phase != URL_OUT) & (is_space | is_newline | ends_html), zero, url_phase)

            markdown_phase = torch.where(is_newline, zero, markdown_phase)
            markdown_table_col = torch.where(is_newline, zero, markdown_table_col)
            markdown_heading_level = torch.where(is_newline, zero, markdown_heading_level)
            markdown_list_depth = torch.where(is_newline, zero, markdown_list_depth)
            markdown_phase = torch.where(line_like_start & is_hash, torch.full_like(markdown_phase, MARKDOWN_HEADING), markdown_phase)
            markdown_heading_level = torch.where(
                line_like_start & is_hash,
                torch.clamp(markdown_heading_level + 1, 0, HEADING_LEVEL_COUNT - 1),
                markdown_heading_level,
            )
            markdown_phase = torch.where(line_like_start & is_dash, torch.full_like(markdown_phase, MARKDOWN_LIST), markdown_phase)
            markdown_list_depth = torch.where(
                line_like_start & is_dash,
                torch.clamp(_len_bucket(indent_len), 0, DEPTH_COUNT - 1),
                markdown_list_depth,
            )
            markdown_phase = torch.where(line_like_start & starts_html, torch.full_like(markdown_phase, MARKDOWN_QUOTE), markdown_phase)
            markdown_phase = torch.where(is_pipe, torch.full_like(markdown_phase, MARKDOWN_TABLE), markdown_phase)
            markdown_table_col = torch.where(is_pipe, markdown_table_col + 1, markdown_table_col)
            markdown_phase = torch.where(quote_kind == QUOTE_BACKTICK, torch.full_like(markdown_phase, MARKDOWN_CODE_FENCE), markdown_phase)
            markdown_fence_state = torch.where(
                line_like_start & (quote_kind == QUOTE_BACKTICK),
                1 - markdown_fence_state,
                markdown_fence_state,
            )

            code_phase = torch.where(is_newline & (bracket_depth == 0), zero, code_phase)
            code_phase = torch.where(is_codeish | (mode == MODE_CODE), torch.full_like(code_phase, CODE_EXPR), code_phase)
            code_phase = torch.where((quote_mode != QUOTE_NONE) & (mode == MODE_CODE), torch.full_like(code_phase, CODE_STRING), code_phase)
            code_phase = torch.where(is_hash & (mode == MODE_CODE), torch.full_like(code_phase, CODE_COMMENT), code_phase)
            code_phase = torch.where(bracket_depth > 0, torch.full_like(code_phase, CODE_BLOCK), code_phase)
            code_call_arg_count = torch.where(opens_round, zero, code_call_arg_count)
            code_call_arg_count = torch.where((bracket_depth > 0) & is_comma, code_call_arg_count + 1, code_call_arg_count)
            code_call_arg_count = torch.where(closes_round, zero, code_call_arg_count)

            prev_abbrev_state = abbrev_state
            dot_in_abbrev = is_dot & (
                (prev_abbrev_state == ABBREV_PENDING_UPPER)
                | (prev_abbrev_state == ABBREV_PENDING_LOWER)
                | (prev_abbrev_state == ABBREV_CHAIN_UPPER)
                | (prev_abbrev_state == ABBREV_CHAIN_LOWER)
            )
            effective_sent_end = is_sent_end & ~dot_in_abbrev
            prose_state = torch.where(effective_sent_end, torch.full_like(prose_state, PROSE_SENTENCE_START), prose_state)
            prose_state = torch.where(is_comma, torch.full_like(prose_state, PROSE_AFTER_PUNCT), prose_state)
            prose_state = torch.where(role == ROLE_DETERMINER, torch.full_like(prose_state, PROSE_AFTER_DETERMINER), prose_state)
            prose_state = torch.where(role == ROLE_PREPOSITION, torch.full_like(prose_state, PROSE_AFTER_PREPOSITION), prose_state)
            prose_state = torch.where((role == ROLE_AUXILIARY) | (role == ROLE_MODAL), torch.full_like(prose_state, PROSE_AFTER_AUX), prose_state)
            subject_like = (role == ROLE_PRONOUN) | (role == ROLE_NOUNISH) | (shape == SHAPE_TITLE)
            prose_state = torch.where(subject_like & (mode == MODE_PLAIN), torch.full_like(prose_state, PROSE_AFTER_SUBJECT), prose_state)
            prose_state = torch.where((role == ROLE_VERBISH) & (mode == MODE_PLAIN), torch.full_like(prose_state, PROSE_AFTER_VERB), prose_state)
            prose_state = torch.where((mode != MODE_PLAIN) & (mode != MODE_MARKDOWN), zero, prose_state)
            prose_clause_stack = torch.where(effective_sent_end | is_newline, zero, prose_clause_stack)
            prose_clause_stack = torch.where(role == ROLE_CONJUNCTION, torch.full_like(prose_clause_stack, PROSE_CLAUSE_SUBORD), prose_clause_stack)
            prose_clause_stack = torch.where(role == ROLE_PREPOSITION, torch.full_like(prose_clause_stack, PROSE_CLAUSE_PP), prose_clause_stack)
            prose_clause_stack = torch.where(
                (role == ROLE_AUXILIARY) | (role == ROLE_MODAL) | (role == ROLE_VERBISH),
                torch.full_like(prose_clause_stack, PROSE_CLAUSE_VP),
                prose_clause_stack,
            )
            prose_clause_stack = torch.where(subject_like | (role == ROLE_DETERMINER), torch.full_like(prose_clause_stack, PROSE_CLAUSE_NP), prose_clause_stack)
            prose_clause_stack = torch.where((mode != MODE_PLAIN) & (mode != MODE_MARKDOWN), zero, prose_clause_stack)

            abbrev_state = torch.where(is_space | is_newline, zero, abbrev_state)
            abbrev_state = torch.where(is_single_upper, torch.full_like(abbrev_state, ABBREV_PENDING_UPPER), abbrev_state)
            abbrev_state = torch.where(is_single_lower, torch.full_like(abbrev_state, ABBREV_PENDING_LOWER), abbrev_state)
            abbrev_state = torch.where(is_dot & (prev_abbrev_state == ABBREV_PENDING_UPPER), torch.full_like(abbrev_state, ABBREV_CHAIN_UPPER), abbrev_state)
            abbrev_state = torch.where(is_dot & (prev_abbrev_state == ABBREV_PENDING_LOWER), torch.full_like(abbrev_state, ABBREV_CHAIN_LOWER), abbrev_state)
            abbrev_state = torch.where(is_word & ~(is_single_upper | is_single_lower), zero, abbrev_state)
            abbrev_state = torch.where(effective_sent_end & ~is_dot, zero, abbrev_state)

            sentence_len = torch.where(effective_sent_end | is_newline, zero, sentence_len + is_tokenish.long())
            clause_reset = effective_sent_end | is_newline | is_comma | is_colon | is_semicolon
            clause_len = torch.where(clause_reset, zero, clause_len + is_tokenish.long())

            serial_edge_state = zero.clone()
            serial_edge_state = torch.where(opens_round | opens_brace | opens_bracket, torch.full_like(serial_edge_state, EDGE_AFTER_OPEN), serial_edge_state)
            serial_edge_state = torch.where(closes_round | closes_brace | closes_bracket, torch.full_like(serial_edge_state, EDGE_AFTER_CLOSE), serial_edge_state)
            serial_edge_state = torch.where(is_comma, torch.full_like(serial_edge_state, EDGE_AFTER_COMMA), serial_edge_state)
            serial_edge_state = torch.where(is_colon, torch.full_like(serial_edge_state, EDGE_AFTER_COLON), serial_edge_state)
            serial_edge_state = torch.where(is_equals, torch.full_like(serial_edge_state, EDGE_AFTER_EQUALS), serial_edge_state)
            serial_edge_state = torch.where(is_scheme_delim, torch.full_like(serial_edge_state, EDGE_AFTER_SCHEME), serial_edge_state)
            serial_edge_state = torch.where(is_dot & ~dot_in_abbrev, torch.full_like(serial_edge_state, EDGE_AFTER_DOT), serial_edge_state)
            serial_edge_state = torch.where(entering_quote, torch.full_like(serial_edge_state, EDGE_AFTER_OPEN_QUOTE), serial_edge_state)
            serial_edge_state = torch.where(leaving_quote, torch.full_like(serial_edge_state, EDGE_AFTER_CLOSE_QUOTE), serial_edge_state)
            followup_edge_state = torch.where(is_layout_token, followup_edge_state, zero)
            followup_edge_state = torch.where(serial_edge_state != EDGE_NONE, serial_edge_state, followup_edge_state)

            active_code_context = (mode == MODE_CODE) | (
                (code_phase != CODE_OUT) & ~protected_scope & (mode != MODE_HTML) & (mode != MODE_JSON) & (mode != MODE_URL)
            )
            next_member_chain = zero.clone()
            next_member_chain = torch.where((mode != MODE_URL) & (mode != MODE_HTML) & (mode != MODE_JSON) & is_dot & ~dot_in_abbrev & ~is_space & ~is_newline, torch.full_like(next_member_chain, CHAIN_AFTER_DOT), next_member_chain)
            next_member_chain = torch.where(active_code_context & opens_round, torch.full_like(next_member_chain, CHAIN_AFTER_CALL_OPEN), next_member_chain)
            next_member_chain = torch.where(active_code_context & opens_bracket, torch.full_like(next_member_chain, CHAIN_AFTER_INDEX_OPEN), next_member_chain)
            next_member_chain = torch.where(active_code_context & (closes_round | closes_bracket), torch.full_like(next_member_chain, CHAIN_AFTER_CLOSE), next_member_chain)
            member_chain_state = next_member_chain

            key_value_side = torch.where(is_bos | is_newline | is_semicolon, zero, key_value_side)
            key_value_side = torch.where((mode == MODE_HTML) & ((html_phase == HTML_OUT) | (html_phase == HTML_TAG_NAME) | (html_phase == HTML_CLOSING)), zero, key_value_side)
            key_value_side = torch.where(html_phase == HTML_ATTR_NAME, torch.full_like(key_value_side, KEYVAL_KEY), key_value_side)
            key_value_side = torch.where((html_phase == HTML_AFTER_EQUALS) | (html_phase == HTML_QUOTED_ATTR), torch.full_like(key_value_side, KEYVAL_VALUE), key_value_side)
            key_value_side = torch.where(opens_json & opens_brace, torch.full_like(key_value_side, KEYVAL_KEY), key_value_side)
            key_value_side = torch.where(opens_json & opens_bracket, torch.full_like(key_value_side, KEYVAL_VALUE), key_value_side)
            key_value_side = torch.where((json_depth > 0) & (json_container_top == JSON_CONTAINER_OBJECT) & is_comma, torch.full_like(key_value_side, KEYVAL_KEY), key_value_side)
            key_value_side = torch.where((json_depth > 0) & (json_container_top == JSON_CONTAINER_ARRAY) & is_comma, torch.full_like(key_value_side, KEYVAL_VALUE), key_value_side)
            key_value_side = torch.where((json_depth > 0) & is_colon, torch.full_like(key_value_side, KEYVAL_VALUE), key_value_side)
            key_value_side = torch.where(active_code_context & (key_value_side == KEYVAL_NONE) & is_tokenish & ~is_equals, torch.full_like(key_value_side, KEYVAL_KEY), key_value_side)
            key_value_side = torch.where(active_code_context & is_equals, torch.full_like(key_value_side, KEYVAL_VALUE), key_value_side)
            key_value_side = torch.where((json_depth == 0) & (mode != MODE_HTML) & (mode != MODE_CODE) & (html_phase == HTML_OUT), zero, key_value_side)
            plain_structured_context = ((mode == MODE_PLAIN) | (mode == MODE_MARKDOWN)) & (markdown_fence_state == 0) & (quote_mode == QUOTE_NONE)
            key_value_side = torch.where(
                plain_structured_context
                & ((current_followup_edge == EDGE_AFTER_COLON) | (current_followup_edge == EDGE_AFTER_EQUALS)),
                torch.full_like(key_value_side, KEYVAL_VALUE),
                key_value_side,
            )

            blank_line = is_newline & (line_token_count == 0)
            blank_line_run = torch.where(blank_line, blank_line_run + 1, blank_line_run)
            blank_line_run = torch.where(is_newline & ~blank_line, zero, blank_line_run)
            line_token_count = torch.where(is_newline, zero, line_token_count)
            line_token_count = torch.where(is_tokenish & ~is_space & ~is_newline, line_token_count + 1, line_token_count)
            curr_indent_bucket = _len_bucket(indent_len)
            line_content_start = line_like_start & is_tokenish & ~is_space & ~is_newline
            indent_delta = torch.where(is_newline, torch.full_like(indent_delta, INDENT_DELTA_NONE), indent_delta)
            indent_delta = torch.where(
                line_content_start & (curr_indent_bucket == last_indent_bucket),
                torch.full_like(indent_delta, INDENT_DELTA_SAME),
                indent_delta,
            )
            indent_delta = torch.where(
                line_content_start & (curr_indent_bucket > last_indent_bucket),
                torch.full_like(indent_delta, INDENT_DELTA_INDENT),
                indent_delta,
            )
            indent_delta = torch.where(
                line_content_start & (curr_indent_bucket < last_indent_bucket),
                torch.full_like(indent_delta, INDENT_DELTA_DEDENT),
                indent_delta,
            )
            last_indent_bucket = torch.where(line_content_start, curr_indent_bucket, last_indent_bucket)
            indent_len = torch.where(is_newline, zero, indent_len)
            indent_len = torch.where(line_like_start & is_space, indent_len + 1, indent_len)
            line_state = torch.where(is_newline, torch.full_like(line_state, LINE_START), line_state)
            line_state = torch.where((line_state == LINE_START) & is_space, torch.full_like(line_state, LINE_AFTER_INDENT), line_state)
            line_state = torch.where(is_word | (~is_space & ~is_newline), zero, line_state)

            scope_state = zero.clone()
            scope_state = torch.where(url_phase != URL_OUT, torch.full_like(scope_state, SCOPE_URL), scope_state)
            scope_state = torch.where((mode == MODE_CODE) & (quote_mode != QUOTE_NONE), torch.full_like(scope_state, SCOPE_CODE_STRING), scope_state)
            scope_state = torch.where(markdown_fence_state > 0, torch.full_like(scope_state, SCOPE_MARKDOWN_CODE), scope_state)
            scope_state = torch.where(html_phase == HTML_QUOTED_ATTR, torch.full_like(scope_state, SCOPE_HTML_ATTR), scope_state)
            scope_state = torch.where((json_depth > 0) & (quote_mode != QUOTE_NONE), torch.full_like(scope_state, SCOPE_JSON_STRING), scope_state)

            if is_bos.any():
                stack_slots = torch.where(is_bos[:, None], torch.zeros_like(stack_slots), stack_slots)
                line_state = torch.where(is_bos, line_start_state, line_state)
                indent_len = torch.where(is_bos, zero, indent_len)
                mode = torch.where(is_bos, zero, mode)
                html_phase = torch.where(is_bos, zero, html_phase)
                json_depth = torch.where(is_bos, zero, json_depth)
                json_expect = torch.where(is_bos, zero, json_expect)
                url_phase = torch.where(is_bos, zero, url_phase)
                markdown_phase = torch.where(is_bos, zero, markdown_phase)
                code_phase = torch.where(is_bos, zero, code_phase)
                quote_mode = torch.where(is_bos, zero, quote_mode)
                scope_state = torch.where(is_bos, zero, scope_state)
                quote_len = torch.where(is_bos, zero, quote_len)
                prose_state = torch.where(is_bos, prose_start_state, prose_state)
                sentence_len = torch.where(is_bos, zero, sentence_len)
                clause_len = torch.where(is_bos, zero, clause_len)
                bracket_depth = torch.where(is_bos, zero, bracket_depth)
                same_class_run = torch.where(is_bos, zero, same_class_run)
                prev_class = torch.where(is_bos, other_class_state, prev_class)
                followup_edge_state = torch.where(is_bos, zero, followup_edge_state)
                stack_top_type = torch.where(is_bos, zero, stack_top_type)
                stack_depth = torch.where(is_bos, zero, stack_depth)
                stack_parent_type = torch.where(is_bos, zero, stack_parent_type)
                html_top_tag_class = torch.where(is_bos, zero, html_top_tag_class)
                html_tag_depth = torch.where(is_bos, zero, html_tag_depth)
                json_container_top = torch.where(is_bos, zero, json_container_top)
                markdown_heading_level = torch.where(is_bos, zero, markdown_heading_level)
                markdown_list_depth = torch.where(is_bos, zero, markdown_list_depth)
                markdown_table_col = torch.where(is_bos, zero, markdown_table_col)
                markdown_fence_state = torch.where(is_bos, zero, markdown_fence_state)
                last_indent_bucket = torch.where(is_bos, zero, last_indent_bucket)
                indent_delta = torch.where(is_bos, zero, indent_delta)
                blank_line_run = torch.where(is_bos, zero, blank_line_run)
                line_token_count = torch.where(is_bos, zero, line_token_count)
                member_chain_state = torch.where(is_bos, zero, member_chain_state)
                key_value_side = torch.where(is_bos, zero, key_value_side)
                code_call_arg_count = torch.where(is_bos, zero, code_call_arg_count)
                prose_clause_stack = torch.where(is_bos, zero, prose_clause_stack)
                serial_edge_state = torch.where(is_bos, zero, serial_edge_state)
                abbrev_state = torch.where(is_bos, zero, abbrev_state)

            features[:, pos, 0] = _len_bucket(indent_len)
            features[:, pos, 1] = mode
            features[:, pos, 2] = html_phase
            features[:, pos, 3] = json_depth
            features[:, pos, 4] = json_expect
            features[:, pos, 5] = url_phase
            features[:, pos, 6] = markdown_phase
            features[:, pos, 7] = code_phase
            features[:, pos, 8] = quote_mode
            features[:, pos, 9] = scope_state
            features[:, pos, 10] = _len_bucket(quote_len)
            features[:, pos, 11] = prose_state
            features[:, pos, 12] = _len_bucket(sentence_len)
            features[:, pos, 13] = _len_bucket(clause_len)
            features[:, pos, 14] = bracket_depth
            features[:, pos, 15] = _run_bucket(same_class_run)
            features[:, pos, 16] = current_followup_edge
            features[:, pos, 17] = stack_top_type
            features[:, pos, 18] = stack_depth
            features[:, pos, 19] = stack_parent_type
            features[:, pos, 20] = html_top_tag_class
            features[:, pos, 21] = html_tag_depth
            features[:, pos, 22] = json_container_top
            features[:, pos, 23] = markdown_heading_level
            features[:, pos, 24] = markdown_list_depth
            features[:, pos, 25] = _len_bucket(markdown_table_col)
            features[:, pos, 26] = markdown_fence_state
            features[:, pos, 27] = indent_delta
            features[:, pos, 28] = _run_bucket(blank_line_run)
            features[:, pos, 29] = current_member_chain
            features[:, pos, 30] = key_value_side
            features[:, pos, 31] = _len_bucket(code_call_arg_count)
            features[:, pos, 32] = prose_clause_stack
            features[:, pos, 33] = serial_edge_state
            features[:, pos, 34] = abbrev_state
        return features

    def embed_feature_ids(self, feature_ids: Tensor) -> Tensor:
        if feature_ids.ndim != 3 or feature_ids.size(-1) != len(FEATURE_SIZES):
            raise ValueError(
                "feature_ids must have shape (batch, time, num_features); "
                f"got {tuple(feature_ids.shape)} expected last_dim={len(FEATURE_SIZES)}"
            )
        out = torch.zeros(
            feature_ids.size(0),
            feature_ids.size(1),
            self.dim,
            device=feature_ids.device,
            dtype=self.embeddings[0].weight.dtype,
        )
        scales = self.feature_scale.to(dtype=out.dtype)
        for idx, emb in enumerate(self.embeddings):
            out = out + scales[idx] * emb(feature_ids[..., idx].long())
        return self.gate.to(dtype=out.dtype) * out

    def forward(self, input_ids: Tensor, feature_ids: Tensor | None = None) -> Tensor:
        if feature_ids is None:
            feature_ids = self.build_feature_ids(input_ids)
        return self.embed_feature_ids(feature_ids)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, features={len(FEATURE_SIZES)}, gate={float(self.gate.detach().cpu()):.4g}"
