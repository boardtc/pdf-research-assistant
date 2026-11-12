"""Regenerate tiny-enthusiasm.pdf, the one-page fixture used by the integration tests.

The PDF is checked in, so this only needs running if its text has to change. It needs
reportlab, which is not a project dependency: `python -m pip install reportlab`, then
run this file. `invariant=1` keeps the output byte-for-byte the same on every run.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A6
from reportlab.pdfgen import canvas

LINES = [
    "Teacher enthusiasm and student learning.",
    "",
    "Students learn more when the teacher",
    "shows genuine enthusiasm for the subject.",
    "Enthusiasm is contagious in a classroom.",
]


def main() -> None:
    output = Path(__file__).with_name("tiny-enthusiasm.pdf")
    pdf = canvas.Canvas(str(output), pagesize=A6, invariant=1)
    pdf.setTitle("Tiny enthusiasm fixture")
    pdf.setAuthor("pdf-research-assistant tests")
    y = 380
    for line in LINES:
        pdf.drawString(20, y, line)
        y -= 16
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
