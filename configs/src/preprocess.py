"""Shared text cleaning for multilingual / African social-media text."""
import re
import html
import unicodedata
from typing import List

URL_RE     = re.compile(r"https?://\S+|www\.\S+")
HTML_RE    = re.compile(r"<[^>]+>")
WS_RE      = re.compile(r"\s+")
EMAIL_RE   = re.compile(r"\S+@\S+\.\S+")
MENTION_RE = re.compile(r"@\w+")

def strip_html(text: str) -> str:
    return HTML_RE.sub(" ", text)

def unescape_html(text: str) -> str:
    return html.unescape(text)

def remove_urls(text: str) -> str:
    return URL_RE.sub(" [URL] ", text)

def remove_emails(text: str) -> str:
    return EMAIL_RE.sub(" [EMAIL] ", text)

def remove_mentions(text: str) -> str:
    return MENTION_RE.sub(" [USER] ", text)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def collapse_ws(text: str) -> str:
    return WS_RE.sub(" ", text).strip()

def clean_text(
    text: str,
    strip_html_flag: bool = True,
    remove_urls_flag: bool = True,
    remove_mentions_flag: bool = True,
    normalize_unicode_flag: bool = True,
    collapse_whitespace_flag: bool = True,
    lowercase: bool = True,
) -> str:
    if not isinstance(text, str):
        return ""
    if strip_html_flag:
        text = unescape_html(text)
        text = strip_html(text)
    text = remove_emails(text)
    if remove_mentions_flag:
        text = remove_mentions(text)
    if remove_urls_flag:
        text = remove_urls(text)
    if normalize_unicode_flag:
        text = normalize_unicode(text)
    if collapse_whitespace_flag:
        text = collapse_ws(text)
    if lowercase:
        text = text.lower()
    return text

def clean_corpus(texts: List[str], **kwargs) -> List[str]:
    return [clean_text(t, **kwargs) for t in texts]
