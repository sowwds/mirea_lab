"""Replace the dimensions on practice 10's drawing with variant 9 values."""

import cairo
from prepare_prac9_pdf import ROOT, Poppler, overlay


SOURCE = ROOT / "prac10/Программирование технологического оборудования. Практические занятия 10.pdf"
OUTPUT = ROOT / "prac10/Практическая 10 — вариант 9.pdf"

# Drawing coordinates: original embedded image, 1338 × 477 pixels.
# Tool diameter, approach/retract clearance and tool-change point stay as in the UP.
LABELS = [
    ((94, 258, 127, 316), "58", True),
    ((21, 191, 118, 227), "21,75±0,1", False),
    ((136, 327, 175, 413), "14,5±0,1", True),
    ((91, 420, 191, 458), "11,625±0,1", False),
    ((312, 420, 414, 458), "23,25±0,1", False),
    ((440, 420, 531, 458), "15,5±0,1", False),
    ((619, 244, 655, 304), "21,75", True),
    ((752, 315, 791, 400), "36,4±0,1", True),
    ((970, 223, 1007, 307), "50,4±0,1", True),
    ((1248, 242, 1283, 301), "56", True),
    ((1017, 420, 1066, 459), "62", False),
]


def main():
    document = Poppler.Document.new_from_file(SOURCE.as_uri(), None)
    surface = cairo.PDFSurface(str(OUTPUT), 595.32, 841.92)
    surface.set_metadata(cairo.PDF_METADATA_TITLE, "Практическая работа 10 — вариант 9")
    ctx = cairo.Context(surface)
    for index in range(document.get_n_pages()):
        page = document.get_page(index)
        surface.set_size(*page.get_size())
        page.render_for_printing(ctx)
        if index == 1:
            overlay(ctx, page.get_image_mapping()[0].area, LABELS, 1338, 477, 37)
        ctx.show_page()
    surface.finish()
    edited = Poppler.Document.new_from_file(OUTPUT.as_uri(), None)
    preview = cairo.ImageSurface(cairo.FORMAT_RGB24, 1470, 550)
    ctx = cairo.Context(preview)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.scale(3, 3)
    ctx.translate(-53, -395)
    edited.get_page(1).render(ctx)
    preview.write_to_png("/tmp/prac10-dimensions-preview.png")
    print(OUTPUT)


if __name__ == "__main__":
    main()
