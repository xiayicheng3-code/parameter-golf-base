from __future__ import annotations

from dataclasses import dataclass
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

ASSIGN_NONE = 0
ASSIGN_LEFT = 1
ASSIGN_RIGHT = 2
ASSIGN_COUNT = 3

PROSE_CLAUSE_NONE = 0
PROSE_CLAUSE_NP = 1
PROSE_CLAUSE_VP = 2
PROSE_CLAUSE_PP = 3
PROSE_CLAUSE_SUBORD = 4
PROSE_CLAUSE_REL = 5
PROSE_CLAUSE_COUNT = 6

FEATURE_NAMES = (
    "line_state",
    "indent_bucket",
    "mode",
    "html_phase",
    "json_depth",
    "json_expect",
    "url_phase",
    "markdown_phase",
    "code_phase",
    "quote_mode",
    "quote_len_bucket",
    "prose_state",
    "sentence_len_bucket",
    "clause_len_bucket",
    "bracket_depth",
    "same_class_run_bucket",
    "punct_state",
    "stack_top_type",
    "stack_depth_bucket",
    "html_top_tag_class",
    "html_tag_depth_bucket",
    "json_container_top",
    "markdown_heading_level",
    "markdown_list_depth",
    "markdown_table_col_bucket",
    "markdown_fence_state",
    "indent_delta",
    "blank_line_run_bucket",
    "code_after_dot",
    "code_assignment_side",
    "code_call_arg_bucket",
    "prose_clause_stack",
)
FEATURE_SIZES = (
    LINE_COUNT,
    LEN_BUCKET_COUNT,
    MODE_COUNT,
    HTML_COUNT,
    DEPTH_COUNT,
    JSON_COUNT,
    URL_COUNT,
    MARKDOWN_COUNT,
    CODE_COUNT,
    QUOTE_COUNT,
    LEN_BUCKET_COUNT,
    PROSE_COUNT,
    LEN_BUCKET_COUNT,
    LEN_BUCKET_COUNT,
    DEPTH_COUNT,
    RUN_BUCKET_COUNT,
    PUNCT_COUNT,
    STACK_COUNT,
    DEPTH_COUNT,
    TAG_COUNT,
    DEPTH_COUNT,
    JSON_CONTAINER_COUNT,
    HEADING_LEVEL_COUNT,
    DEPTH_COUNT,
    LEN_BUCKET_COUNT,
    BOOL_COUNT,
    INDENT_DELTA_COUNT,
    RUN_BUCKET_COUNT,
    BOOL_COUNT,
    ASSIGN_COUNT,
    LEN_BUCKET_COUNT,
    PROSE_CLAUSE_COUNT,
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


def _flags(text: str, role: int) -> int:
    flags = 0
    if text.isspace() or " " in text or "\t" in text:
        flags |= F_SPACE
    if "\n" in text or "\r" in text:
        flags |= F_NEWLINE
    if any(ch in text for ch in ".!?"):
        flags |= F_SENT_END
    if any(ch in text for ch in ",;:"):
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
        flags[token_id] = _flags(text, role)
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
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=dim ** -0.5)

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
        quote_len = torch.zeros(batch, device=device, dtype=torch.long)
        prose_state = torch.full((batch,), PROSE_SENTENCE_START, device=device, dtype=torch.long)
        sentence_len = torch.zeros(batch, device=device, dtype=torch.long)
        clause_len = torch.zeros(batch, device=device, dtype=torch.long)
        bracket_depth = torch.zeros(batch, device=device, dtype=torch.long)
        same_class_run = torch.zeros(batch, device=device, dtype=torch.long)
        prev_class = torch.full((batch,), TOKEN_OTHER, device=device, dtype=torch.long)
        punct_state = torch.zeros(batch, device=device, dtype=torch.long)
        stack_top_type = torch.zeros(batch, device=device, dtype=torch.long)
        stack_depth = torch.zeros(batch, device=device, dtype=torch.long)
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
        code_after_dot = torch.zeros(batch, device=device, dtype=torch.long)
        code_assignment_side = torch.zeros(batch, device=device, dtype=torch.long)
        code_call_arg_count = torch.zeros(batch, device=device, dtype=torch.long)
        prose_clause_stack = torch.zeros(batch, device=device, dtype=torch.long)
        zero = torch.zeros(batch, device=device, dtype=torch.long)

        for pos in range(steps):
            tok = ids[:, pos]
            tok_class = self.token_class_lut[tok]
            role = self.lexical_role_lut[tok]
            shape = self.shape_lut[tok]
            quote_kind = self.quote_kind_lut[tok]
            tag_class = self.tag_class_lut[tok]
            flags = self.flags_lut[tok]

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
            is_word = (tok_class == TOKEN_ALPHA) | (tok_class == TOKEN_DIGIT)
            is_tokenish = ~(is_space | is_newline)
            line_like_start = (line_state == LINE_START) | (line_state == LINE_AFTER_INDENT)
            prev_html_phase = html_phase

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
            stack_depth = torch.clamp(stack_depth + any_struct_open.long() - any_struct_close.long(), 0, DEPTH_COUNT - 1)
            stack_top_type = torch.where(stack_depth == 0, zero, stack_top_type)
            stack_top_type = torch.where(opens_round, torch.full_like(stack_top_type, STACK_PAREN), stack_top_type)
            jsonish_context = (mode == MODE_JSON) | (json_depth > 0)
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
            stack_top_type = torch.where(opens_bracket, bracket_top, stack_top_type)
            stack_top_type = torch.where(opens_brace, brace_top, stack_top_type)
            stack_top_type = torch.where(starts_html, torch.full_like(stack_top_type, STACK_HTML), stack_top_type)
            stack_top_type = torch.where(quote_mode != QUOTE_NONE, torch.full_like(stack_top_type, STACK_QUOTE), stack_top_type)

            mode = torch.where((mode == MODE_URL) & (is_space | is_newline | ends_html), zero, mode)
            mode = torch.where(starts_html, torch.full_like(mode, MODE_HTML), mode)
            mode = torch.where(is_urlish & ~is_space & ~is_newline, torch.full_like(mode, MODE_URL), mode)
            mode = torch.where((json_depth > 0) & (mode == MODE_PLAIN), torch.full_like(mode, MODE_JSON), mode)
            mode = torch.where((json_depth == 0) & (mode == MODE_JSON), zero, mode)
            mode = torch.where(is_markdown & line_like_start & (mode == MODE_PLAIN), torch.full_like(mode, MODE_MARKDOWN), mode)
            mode = torch.where(is_codeish & (mode == MODE_PLAIN), torch.full_like(mode, MODE_CODE), mode)
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
            json_expect = torch.where((json_depth > 0) & is_colon, torch.full_like(json_expect, JSON_EXPECT_VALUE), json_expect)
            json_expect = torch.where((json_depth > 0) & is_comma, torch.full_like(json_expect, JSON_EXPECT_KEY), json_expect)
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
            url_phase = torch.where((url_phase != URL_OUT) & (tok_class == TOKEN_SLASH), torch.full_like(url_phase, URL_PATH), url_phase)
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
            code_after_dot = torch.where(is_dot, torch.ones_like(code_after_dot), code_after_dot)
            code_after_dot = torch.where(is_word & (code_after_dot > 0), zero, code_after_dot)
            code_assignment_side = torch.where(is_newline | is_semicolon, zero, code_assignment_side)
            code_assignment_side = torch.where(
                (code_phase != CODE_OUT) & is_tokenish & (code_assignment_side == ASSIGN_NONE),
                torch.full_like(code_assignment_side, ASSIGN_LEFT),
                code_assignment_side,
            )
            code_assignment_side = torch.where(is_equals, torch.full_like(code_assignment_side, ASSIGN_RIGHT), code_assignment_side)
            code_call_arg_count = torch.where(opens_round, zero, code_call_arg_count)
            code_call_arg_count = torch.where((bracket_depth > 0) & is_comma, code_call_arg_count + 1, code_call_arg_count)
            code_call_arg_count = torch.where(closes_round, zero, code_call_arg_count)

            prose_state = torch.where(is_sent_end, torch.full_like(prose_state, PROSE_SENTENCE_START), prose_state)
            prose_state = torch.where(is_comma, torch.full_like(prose_state, PROSE_AFTER_PUNCT), prose_state)
            prose_state = torch.where(role == ROLE_DETERMINER, torch.full_like(prose_state, PROSE_AFTER_DETERMINER), prose_state)
            prose_state = torch.where(role == ROLE_PREPOSITION, torch.full_like(prose_state, PROSE_AFTER_PREPOSITION), prose_state)
            prose_state = torch.where((role == ROLE_AUXILIARY) | (role == ROLE_MODAL), torch.full_like(prose_state, PROSE_AFTER_AUX), prose_state)
            subject_like = (role == ROLE_PRONOUN) | (role == ROLE_NOUNISH) | (shape == SHAPE_TITLE)
            prose_state = torch.where(subject_like & (mode == MODE_PLAIN), torch.full_like(prose_state, PROSE_AFTER_SUBJECT), prose_state)
            prose_state = torch.where((role == ROLE_VERBISH) & (mode == MODE_PLAIN), torch.full_like(prose_state, PROSE_AFTER_VERB), prose_state)
            prose_state = torch.where((mode != MODE_PLAIN) & (mode != MODE_MARKDOWN), zero, prose_state)
            prose_clause_stack = torch.where(is_sent_end | is_newline, zero, prose_clause_stack)
            prose_clause_stack = torch.where(role == ROLE_CONJUNCTION, torch.full_like(prose_clause_stack, PROSE_CLAUSE_SUBORD), prose_clause_stack)
            prose_clause_stack = torch.where(role == ROLE_PREPOSITION, torch.full_like(prose_clause_stack, PROSE_CLAUSE_PP), prose_clause_stack)
            prose_clause_stack = torch.where(
                (role == ROLE_AUXILIARY) | (role == ROLE_MODAL) | (role == ROLE_VERBISH),
                torch.full_like(prose_clause_stack, PROSE_CLAUSE_VP),
                prose_clause_stack,
            )
            prose_clause_stack = torch.where(subject_like | (role == ROLE_DETERMINER), torch.full_like(prose_clause_stack, PROSE_CLAUSE_NP), prose_clause_stack)
            prose_clause_stack = torch.where((mode != MODE_PLAIN) & (mode != MODE_MARKDOWN), zero, prose_clause_stack)

            sentence_len = torch.where(is_sent_end | is_newline, zero, sentence_len + is_tokenish.long())
            clause_reset = is_sent_end | is_newline | is_comma | is_colon | is_semicolon
            clause_len = torch.where(clause_reset, zero, clause_len + is_tokenish.long())

            punct_state = torch.where(is_word, zero, punct_state)
            punct_state = torch.where(opens_json | opens_paren, torch.full_like(punct_state, PUNCT_OPEN), punct_state)
            punct_state = torch.where(is_comma, torch.full_like(punct_state, PUNCT_COMMA), punct_state)
            punct_state = torch.where(is_colon, torch.full_like(punct_state, PUNCT_COLON), punct_state)
            punct_state = torch.where(is_semicolon, torch.full_like(punct_state, PUNCT_SEMICOLON), punct_state)
            punct_state = torch.where(is_sent_end, torch.full_like(punct_state, PUNCT_SENTENCE), punct_state)

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

            features[:, pos, 0] = line_state
            features[:, pos, 1] = _len_bucket(indent_len)
            features[:, pos, 2] = mode
            features[:, pos, 3] = html_phase
            features[:, pos, 4] = json_depth
            features[:, pos, 5] = json_expect
            features[:, pos, 6] = url_phase
            features[:, pos, 7] = markdown_phase
            features[:, pos, 8] = code_phase
            features[:, pos, 9] = quote_mode
            features[:, pos, 10] = _len_bucket(quote_len)
            features[:, pos, 11] = prose_state
            features[:, pos, 12] = _len_bucket(sentence_len)
            features[:, pos, 13] = _len_bucket(clause_len)
            features[:, pos, 14] = bracket_depth
            features[:, pos, 15] = _run_bucket(same_class_run)
            features[:, pos, 16] = punct_state
            features[:, pos, 17] = stack_top_type
            features[:, pos, 18] = stack_depth
            features[:, pos, 19] = html_top_tag_class
            features[:, pos, 20] = html_tag_depth
            features[:, pos, 21] = json_container_top
            features[:, pos, 22] = markdown_heading_level
            features[:, pos, 23] = markdown_list_depth
            features[:, pos, 24] = _len_bucket(markdown_table_col)
            features[:, pos, 25] = markdown_fence_state
            features[:, pos, 26] = indent_delta
            features[:, pos, 27] = _run_bucket(blank_line_run)
            features[:, pos, 28] = code_after_dot
            features[:, pos, 29] = code_assignment_side
            features[:, pos, 30] = _len_bucket(code_call_arg_count)
            features[:, pos, 31] = prose_clause_stack
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
