"""Regenerate the tiny synthetic cards.png.

Uses Pillow at author-time only. Pillow is NOT a workspace runtime or
test dep -- install ad-hoc:

    uv run --with pillow python examples/build_example_image.py

The output is committed as a binary. Two cards are the "same" person
(same email, slightly different name spelling) so the dedup demo
has something to demonstrate.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent
_OUT = _HERE / "cards.png"

CARDS = [
    {
        "name": "Alice Chen",
        "title": "Head of Product",
        "company": "Northwind Analytics",
        "email": "alice@northwind.io",
        "phone": "+1 415-555-0101",
    },
    {
        "name": "Alicia Chen",
        "title": "",
        "company": "Northwind Analytics",
        "email": "alice@northwind.io",
        "phone": "",
    },
    {
        "name": "Bob Rivera",
        "title": "Backend Engineer",
        "company": "Ridgeway Robotics",
        "email": "bob@ridgeway.dev",
        "phone": "+1 415-555-0199",
    },
]


def main() -> int:
    from PIL import Image, ImageDraw, ImageFont

    card_w, card_h = 480, 140
    gap = 20
    total_h = 3 * card_h + 4 * gap
    img = Image.new("RGB", (card_w + 2 * gap, total_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    try:
        font_name = ImageFont.truetype("arial.ttf", 22)
        font_body = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_name = ImageFont.load_default()
        font_body = ImageFont.load_default()

    for i, card in enumerate(CARDS):
        y0 = gap + i * (card_h + gap)
        draw.rectangle(
            [(gap, y0), (gap + card_w, y0 + card_h)],
            fill=(255, 255, 255),
            outline=(80, 80, 80),
            width=2,
        )
        draw.text((gap + 16, y0 + 12), card["name"], font=font_name, fill=(20, 20, 20))
        y = y0 + 46
        for key in ("title", "company", "email", "phone"):
            if card[key]:
                draw.text((gap + 16, y), card[key], font=font_body, fill=(60, 60, 60))
                y += 20

    img.save(_OUT, format="PNG")
    print(f"wrote {_OUT} ({_OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
