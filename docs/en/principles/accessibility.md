# Accessibility

## Never colour alone

Every series carries a redundant encoding - a dash pattern or a marker shape -
alongside its colour. An assumption's status is shown three ways at once: an
icon, a word, and a colour.

All seven themes meet WCAG AA contrast for body text. `high_contrast` reaches
21:1. A colourblind-safe palette is included, and the contrast of every theme
is checked by the test suite rather than by eye.

## Motion

Reduced motion is a rule, not a hint. When it is on:

- transitions and animations are disabled outright in the stylesheet;
- autoplay is refused, and says why;
- every frame stays reachable by hand, with its full explanation.

The application also honours the operating system's `prefers-reduced-motion`
setting without being asked.

## Text alternatives

Every figure carries a description built from what the lab said about it -
its title, its caption, its evidence type and the headline number - rather
than the word "chart".

Assumptions expose ARIA status attributes; warnings are announced as alerts.

## Keyboard

Every control is reachable by keyboard with a visible focus ring. The shortcut
map lives in one place in the code, so the help dialog and the handlers cannot
disagree; press the keyboard icon in the header to see it.

## Language and direction

Arabic is right-to-left by architecture rather than by a stylesheet patch: the
document direction, the layout mirroring, the font stack and the isolation of
Latin identifiers inside Arabic prose are all handled at the source.
