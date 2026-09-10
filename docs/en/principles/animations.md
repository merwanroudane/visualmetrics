# Animations must explain themselves

An animation is the most persuasive thing a teaching tool can show, and the
easiest to misread. A cloud of points converging looks like a proof of
convergence. It is not one.

So every animation in VisualMetrics carries an explanation layer, and every
frame of it answers the same questions:

| Field | The question it answers |
|---|---|
| `what_you_see` | What is on screen right now |
| `what_changed` | What moved since the previous frame |
| `why` | Why it moved - the mechanism, not the description |
| `interpretation` | How to read what just happened |
| `conclusion` | What you may take away |
| `warning` | What you must not take away |
| `math` | The equation this frame illustrates |
| `outputs` | The numbers at this frame |

The animation as a whole states its `purpose` before it starts and its
`summary` after it ends. Each frame may also name the assumptions in force and
those violated at that moment.

## This is enforced

- an animation with no annotated steps is **refused** by the exporter and by
  the GUI player, with a visible message rather than a silent picture;
- the test suite checks every frame of every animation in the catalogue for
  the four required fields;
- exporting an animation to a static format is refused, because the export
  would lose the commentary. `export_animation_frames` writes the images *and*
  a JSON file of the commentary beside them.

## Reduced motion

Motion is a preference the application obeys. With reduced motion on,
autoplay is refused - with an explanation, rather than silently ignored - and
every frame remains reachable by hand with its explanation intact.

Nothing is only available to someone who can watch it move.
