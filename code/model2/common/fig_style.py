"""Shared title hierarchy for manuscript figures.

Sizes are defined for a REF_WIDTH_IN-wide figure and scaled by the actual figure
width, so titles print at the same size once every figure is fitted to the page.
Text is sentence case; panel titles read "a  Title".
"""
from __future__ import annotations

REF_WIDTH_IN = 12.0
SUPTITLE_PT = 15.0
PANEL_PT = 12.0
SUBTITLE_PT = 9.0
TITLE_COLOR = "0.10"
SUBTITLE_COLOR = "0.35"


def _scale(fig) -> float:
    return fig.get_figwidth() / REF_WIDTH_IN


def suptitle_pt(fig) -> float:
    return SUPTITLE_PT * _scale(fig)


def panel_pt(fig) -> float:
    return PANEL_PT * _scale(fig)


def subtitle_pt(fig) -> float:
    return SUBTITLE_PT * _scale(fig)


def add_suptitle(fig, text: str, y: float = 0.98, **kw):
    return fig.suptitle(
        text, fontsize=suptitle_pt(fig), fontweight="bold", color=TITLE_COLOR, y=y, **kw,
    )


def panel_label(letter: str | None, text: str) -> str:
    return f"{letter.lower()}  {text}" if letter else text


def panel_title(ax, letter: str | None, text: str, pad: float = 6.0, **kw):
    return ax.set_title(
        panel_label(letter, text), loc="left", fontsize=panel_pt(ax.figure),
        fontweight="bold", color=TITLE_COLOR, pad=pad, **kw,
    )


def subtitle_kw(fig) -> dict:
    return dict(fontsize=subtitle_pt(fig), fontweight="normal", color=SUBTITLE_COLOR)


def panel_kw(fig) -> dict:
    return dict(fontsize=panel_pt(fig), fontweight="bold", color=TITLE_COLOR)
