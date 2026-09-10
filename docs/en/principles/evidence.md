# Evidence types

This is the idea the package rests on.

A picture that illustrates a theorem and a Monte Carlo study that suggests a
pattern are not the same kind of claim. Presenting them the same way teaches
students that they are - which is the single most damaging thing a statistics
tool can do.

So every figure, animation and lab declares what kind of evidence it offers.

## The nine types

| Type | Icon | What it claims |
|---|:--:|---|
| `formal_proof` | 📐 | A complete mathematical argument, not a picture of one. |
| `symbolic_derivation` | ∑ | Each algebraic step is shown and can be checked. |
| `geometric_proof` | ⊥ | The geometry is exact: the drawn relation is the theorem. |
| `visual_derivation` | ✎ | A faithful visual account of a valid derivation, not a substitute for it. |
| `visual_intuition` | 👁 | Intuition only. It suggests why a result is plausible; it proves nothing. |
| `simulation` | 🎲 | Finite Monte Carlo evidence. A simulation is never a proof of a theorem. |
| `numerical_demonstration` | # | Exact computation for the displayed parameter values only. |
| `counterexample` | ⚠ | One valid counterexample is enough to refute a universal claim. |
| `empirical_example` | 📊 | Real data illustrate; they do not verify a theorem. |

Three of them - `formal_proof`, `symbolic_derivation` and `geometric_proof` -
count as proof. The other six do not, and no amount of visual polish promotes
them.

## What is enforced

These are tests, not intentions:

- a `ProofSpec` refuses any evidence type that is not one of the three proof
  kinds;
- a proof will not construct without a statement of what it does **not**
  establish;
- a lab may not carry a proof badge unless it points at a real proof in the
  library;
- a numerical check attached to a proof is reported as
  `numerical_demonstration` and described as a test of the code, never as
  evidence for the theorem;
- the badge and its caveat travel together into every export.

## Why a check is not evidence

Several proofs ship a numerical check - the Frisch-Waugh-Lovell theorem
compares the two coefficient vectors, the projection proof samples the subspace
and confirms nothing is closer.

These catch **implementation drift** between the proof text and the code that
illustrates it. They are not evidence for the theorem: the theorem is already
proved, and a finite computation could not establish it in any case. The
interface says so wherever the checks appear.

## Reading a badge

A badge is a claim about the *strongest* thing the figure supports. If a lab
shows you a simulation with 10,000 replications and the rejection rate sits at
13.8% against a nominal 5%, that is strong evidence about this data-generating
process at this sample size. It is not a theorem about all of them.
