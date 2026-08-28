# Example image

`cards.png` is a synthetic 3-card photo generated at author-time by
`build_example_image.py` using ONLY Pillow (added ad-hoc via
`uv run --with pillow`; Pillow is NOT a runtime or test dep, matching
#11's `reportlab` pattern). All names, companies, emails, and phone
numbers are self-authored fakes -- no real people, no real companies,
no third-party rights.

Layout: two of the three "cards" are the same person with slightly
different name spellings and the same email, so the deduplication
demo has something to demonstrate:

    +--------------------+
    | Alice Chen         |
    | alice@northwind.io |
    +--------------------+
    +--------------------+
    | Alicia Chen        |
    | alice@northwind.io |
    +--------------------+
    +--------------------+
    | Bob Rivera         |
    | bob@ridgeway.dev   |
    +--------------------+

Point `--image` at your own photo (any `.png/.jpg/.jpeg/.webp` under
20 MB) to try the extractor on real data.
