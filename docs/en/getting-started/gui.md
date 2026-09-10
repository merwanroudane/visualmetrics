# The GUI

```bash
visualmetrics
```

Then open <http://localhost:8080>.

## The pages

| Route | What it holds |
|---|---|
| `/` | Overview, counts, what the badges mean |
| `/catalog` | Every concept, searchable in three languages |
| `/lab/<concept_id>` | A lab: controls, scenarios, knowledge tabs, export |
| `/proofs` and `/proof/<id>` | The proof library and the step viewer |
| `/glossary` | Every term in three languages |
| `/paths` | Curated learning paths |
| `/doctor` | What is installed, and what each gap disables |

## Inside a lab

The **left panel** holds the scenarios, grouped by what they are for, and the
controls the lab declares. Scenarios come first on purpose: the interesting
question is usually "what happens when this assumption fails", not "what
happens at 0.37".

The **knowledge tabs** are built from what the lab produced - overview,
intuition, visualize, animate, simulation, diagnostics, compare,
counterexample, assumptions, mathematics, proof, interpretation, common
mistakes, warnings, data, code, self-check and references. A tab appears only
when there is something in it.

Above them sits the **evidence badge** with its caveat, and any warning the run
raised. That order is deliberate: you learn what kind of claim you are looking
at before you look at it.

## Switching language, theme and level

The header switches are available on every page and take effect **in place** -
you keep your figure, your parameters and your scroll position. A teacher can
switch to Arabic mid-explanation without losing the class's place.

- **Language**: English, العربية, Français. Arabic renders right-to-left
  throughout, with Latin identifiers isolated so they read correctly inside
  Arabic prose.
- **Theme**: seven themes, including high-contrast and colourblind-safe. The
  page and the figures use the same palette, so an exported chart matches the
  page it came from.
- **Level**: beginner, intermediate, advanced, PhD. This changes what is shown,
  not only how much.
- **Terminology**: translated, bilingual, or English technical terms in
  translated prose.

## Reduced motion and presentation mode

Reduced motion is obeyed as a rule, not a hint: transitions are disabled,
autoplay is refused, and every animation frame stays reachable by hand with its
explanation intact.

Presentation mode enlarges the type and hides the controls, for showing a lab
to a room.

## Sharing and exporting

**Copy link** encodes the session - language, theme, scenario, seed and every
parameter - so a colleague opening it sees your figure, not their defaults.

**Export** writes a self-contained HTML report carrying the evidence badge, the
assumptions, the warnings, the numbers, the generated Python and the frame
commentary. There is no export that strips them.
