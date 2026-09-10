# Working in three languages

VisualMetrics is trilingual by architecture, not by translation patch. The
interface, the catalogue, the glossary, the proof structure, the errors and the
scenario names all exist in English, Arabic and French, and a test fails if a
key exists in one language and not the others.

## Switching

```python
import visualmetrics as vm

vm.configure(language="ar")
vm.lab("inference.clt", language="fr")
```

In the GUI, use the header selector; it takes effect in place.

## Terminology modes

A class that will go on to read English papers needs the English term even when
working in Arabic. So technical vocabulary can be shown three ways:

| Mode | What you see |
|---|---|
| `translated` | The translated term only |
| `bilingual` | The translated term with the English one beside it |
| `english_technical` | Prose in your language, technical terms in English |

The glossary drives all three, so the choice is consistent everywhere.

## Arabic

- The document direction, the layout and the figure axes all mirror.
- Latin identifiers inside Arabic prose are wrapped in Unicode isolates, so
  `R-squared` does not scramble the sentence around it.
- Search normalises Arabic orthography - diacritics, tatweel, and the alef,
  yeh and teh-marbuta variants - so a term found without diacritics still
  matches.
- The font stack asks for a real Naskh face rather than falling back to a Latin
  font.

## The honest gap

The interface is fully translated. The **lab narrative** is not: it uses about
2,400 translation keys, and only the shared vocabulary is translated so far, so
Arabic and French currently fall back to English inside a lab.

The fallback is deliberate - showing a raw key would be worse - and the missing
keys are recorded rather than silently swallowed. See
[Adding a translation](contributing.md).
