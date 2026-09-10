# Adding a translation

This is the most valuable contribution available to the project right now.

## Where the text lives

```
src/visualmetrics/i18n/
    en/  common.json  concepts.json  labs.json  proofs.json  glossary.json  errors.json
    ar/  ...
    fr/  ...
```

Keys must match English exactly. A test compares the three bundles and fails on
any difference, in either direction - a key in Arabic that English does not
have is also an error, because it means something was renamed.

## The open work

`labs.json`. The lab narrative uses roughly 2,400 keys and only the shared
vocabulary is translated. Translating a single lab is a complete and useful
contribution: pick a lab, collect its `labs.<lab>.*` keys, and add them to
`ar` and `fr`.

## Guidance

- Keep technical terms consistent with `glossary.json`; it drives the three
  terminology modes, so an inconsistency there shows up everywhere.
- In Arabic, leave Latin identifiers as they are - the renderer isolates them.
- Translate the *meaning*. Several strings explain a subtlety, and a literal
  rendering often loses it. If a sentence does not work in your language,
  rewrite it so that it does.
- Explanations of misconceptions are the hardest and the most important: the
  wrong belief has to sound plausible in your language, or the correction has
  nothing to correct.

## A new language

1. Add `src/visualmetrics/i18n/<code>/` with the six bundles.
2. Register the code in `i18n/translator.py`.
3. If it is right-to-left, add it in `i18n/rtl.py`.
4. Add a font stack in `gui/themes/css.py` if the default does not suit it.
