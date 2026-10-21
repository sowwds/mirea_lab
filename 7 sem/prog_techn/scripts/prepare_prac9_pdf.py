"""Overlay variant 9 dimensions on the original drawing, preserving its lines."""

from pathlib import Path
import math

import cairo
import gi

gi.require_version("Poppler", "0.18")
from gi.repository import Poppler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "prac9/Программирование технологического оборудования. Практические занятия 09.pdf"
OUTPUT = ROOT / "prac9/Практическая 09 — вариант 9.pdf"

# Bounds use the embedded drawing's original 1133 × 765 pixel coordinate space.
# Existing manufacturing tolerances are retained, while nominal dimensions change.
LABELS = [
    ((408, 149, 444, 179), "41", False),
    ((61, 127, 114, 157), "R16,4", False),
    ((461, 235, 512, 264), "R16,4", False),
    ((239, 349, 286, 380), "R12,3", False),
    ((259, 439, 306, 469), "R12,3", False),
    ((291, 387, 342, 418), "R24,6", False),
    ((365, 541, 404, 570), "R6,56", False),
    ((295, 718, 352, 749), "82", False),
    ((0, 383, 28, 445), "82", True),
    ((49, 373, 80, 450), "57,4±0,1", True),
    ((49, 633, 80, 706), "12,3±0,1", True),
    ((547, 673, 617, 705), "12,3±0,1", False),
    ((287, 673, 362, 705), "57,4±0,1", False),
    ((549, 542, 579, 608), "17,48", True),
    ((549, 449, 579, 511), "18,87", True),
    ((549, 321, 579, 377), "6,41", True),
    ((235, 626, 286, 658), "6,41", False),
    ((363, 626, 426, 659), "18,87", False),
    ((452, 626, 512, 659), "17,48", False),
    ((842, 643, 921, 674), "26±0,1", False),
    ((875, 686, 926, 717), "39", False),
    ((976, 384, 1009, 416), "13", False),
]


def overlay(ctx, image_rect, labels=None, width=1133, height=765, font_size=31):
    if labels is None:
        labels = LABELS
    ctx.save()
    ctx.translate(image_rect.x1, image_rect.y1)
    ctx.scale((image_rect.x2 - image_rect.x1) / width,
              (image_rect.y2 - image_rect.y1) / height)
    for (x0, y0, x1, y1), text, vertical in labels:
        ctx.set_source_rgb(1, 1, 1)
        ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
        ctx.fill()
        ctx.save()
        ctx.translate((x0 + x1) / 2, (y0 + y1) / 2)
        if vertical:
            ctx.rotate(-math.pi / 2)
        width, height = (y1 - y0, x1 - x0) if vertical else (x1 - x0, y1 - y0)
        ctx.select_font_face("Fira Sans Condensed ExtraLight",
                             cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(font_size)
        ext = ctx.text_extents(text)
        sx = min(1.0, (width - 2) / ext.width)
        sy = min(1.0, (height - 4) / ext.height)
        ctx.scale(sx, sy)
        ctx.move_to(-ext.x_bearing - ext.width / 2,
                    -ext.y_bearing - ext.height / 2)
        ctx.set_source_rgb(0, 0, 0)
        ctx.show_text(text)
        ctx.restore()
    ctx.restore()


def main():
    document = Poppler.Document.new_from_file(SOURCE.as_uri(), None)
    surface = cairo.PDFSurface(str(OUTPUT), 595.32, 841.92)
    surface.set_metadata(cairo.PDF_METADATA_TITLE, "Практическая работа 9 — вариант 9")
    ctx = cairo.Context(surface)
    for index in range(document.get_n_pages()):
        page = document.get_page(index)
        width, height = page.get_size()
        surface.set_size(width, height)
        page.render_for_printing(ctx)
        if index == 1:
            overlay(ctx, page.get_image_mapping()[0].area)
        ctx.show_page()
    surface.finish()
    # A render of the edited page is used for visual inspection.
    edited = Poppler.Document.new_from_file(OUTPUT.as_uri(), None)
    preview = cairo.ImageSurface(cairo.FORMAT_RGB24, 1230, 850)
    ctx = cairo.Context(preview)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.scale(3, 3)
    ctx.translate(-91, -396)
    edited.get_page(1).render(ctx)
    preview.write_to_png("/tmp/prac9-dimensions-preview.png")
    print(OUTPUT)


if __name__ == "__main__":
    main()
