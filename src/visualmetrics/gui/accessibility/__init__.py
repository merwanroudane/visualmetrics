"""Accessibility helpers: text alternatives, contrast, motion and keyboard map."""

from .a11y import (
    announce,
    aria_for_assumption,
    contrast_ratio,
    describe_series,
    figure_description,
    keyboard_shortcuts,
    meets_contrast,
)

__all__ = [
    "figure_description",
    "describe_series",
    "keyboard_shortcuts",
    "aria_for_assumption",
    "announce",
    "contrast_ratio",
    "meets_contrast",
]
