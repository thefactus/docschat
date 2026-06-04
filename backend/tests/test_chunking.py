"""
Tests for _split and _parse. Both are pure/deterministic: no DB, no API calls.
All assertions use the live settings values (chunk_size=1500, chunk_overlap=300)
so they stay in sync with the config.
"""


from app.config import settings
from app.ingestion.pipeline import _parse, _split

# ---------------------------------------------------------------------------
# _split
# ---------------------------------------------------------------------------


def test_oversized_single_paragraph_splits_at_sentence_boundary():
    # Build one paragraph > chunk_size with clear sentence boundaries.
    # At ~50 chars/sentence, 40 sentences = ~2000 chars — comfortably over 1500.
    sentences = [f"Sentence number {i} ends here with a period." for i in range(40)]
    text = " ".join(sentences)
    assert len(text) > settings.chunk_size

    chunks = _split(text)

    assert len(chunks) >= 2
    # Each chunk must not exceed chunk_size (the split searches backward from the
    # limit, so the last resort ' ' split may leave a chunk at exactly chunk_size).
    for chunk in chunks:
        assert len(chunk) <= settings.chunk_size, f"Chunk too long: {len(chunk)}"
    assert all(len(c) > 0 for c in chunks)


def test_overlap_carries_tail_of_previous_chunk():
    # Two paragraphs that together exceed chunk_size, forcing a split.
    # At ~800 chars each, combined ~1602 > 1500, so they must split.
    para1 = ("Alpha content fills up the first paragraph block. " * 16).strip()  # ~800 chars
    para2 = ("Beta content fills up the second paragraph block. " * 16).strip()  # ~800 chars
    assert len(para1) + len(para2) > settings.chunk_size

    chunks = _split(f"{para1}\n\n{para2}")

    assert len(chunks) >= 2
    # The overlap tail (last chunk_overlap chars of chunk[0]) must appear in chunk[1].
    tail = chunks[0][-settings.chunk_overlap :]
    first_word = tail.strip().split()[0]
    assert first_word in chunks[1], (
        f"Overlap tail not carried forward.\n"
        f"Expected '{first_word}' near start of chunk[1]."
    )


def test_tiny_fragments_are_dropped():
    # Fragments <= 30 chars are filtered by _split.
    text = (
        "Hi.\n\nHello.\n\n"
        "This paragraph is definitely long enough to survive the minimum length filter."
    )
    chunks = _split(text)
    for chunk in chunks:
        assert len(chunk) > 30, f"Short fragment leaked through: {repr(chunk)}"
    assert any("survive" in c for c in chunks)


def test_empty_input_yields_no_chunks():
    assert _split("") == []


def test_whitespace_only_input_yields_no_chunks():
    assert _split("   \n\n   \t\n") == []


def test_single_short_paragraph_within_limit():
    text = "This is a short document that fits entirely within one chunk."
    chunks = _split(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_multiple_small_paragraphs_packed_into_one_chunk():
    # Small paragraphs well under chunk_size should be packed together, not fragmented.
    paras = "\n\n".join([f"Short paragraph {i}." for i in range(10)])
    assert len(paras) < settings.chunk_size
    chunks = _split(paras)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------


def test_parse_plain_text_returns_single_pair_with_no_page():
    content = b"Hello world. This is a plain text document."
    result = _parse(content, ".txt")
    assert len(result) == 1
    page, text = result[0]
    assert page is None
    assert "Hello world" in text


def test_parse_markdown_returns_single_pair_with_no_page():
    content = b"# Title\n\nSome markdown content here."
    result = _parse(content, ".md")
    assert len(result) == 1
    page, text = result[0]
    assert page is None


def test_parse_plain_text_utf8_decoding():
    content = "Ação, naïve, façade.".encode("utf-8")
    result = _parse(content, ".txt")
    assert "Ação" in result[0][1]


def test_parse_empty_text_file_returns_one_pair():
    content = b""
    result = _parse(content, ".txt")
    assert len(result) == 1
    assert result[0][0] is None
