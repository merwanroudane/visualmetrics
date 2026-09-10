# Scenarios

Every lab exposes named scenarios grouped by what they are for. The point is
not convenience: a lab that only shows the case where a method works teaches
the wrong lesson, so at least one failure case is required of every lab.

Across the 47 built labs there are
519 scenarios.

| Category | What it is for | Count |
|---|---|---:|
| `compare_methods` | Compare methods | 85 |
| `canonical` | The standard case | 54 |
| `boundary` | At the boundary | 43 |
| `violation` | Assumption violated | 43 |
| `counterexample` | Counterexample | 41 |
| `small_sample` | Small sample | 39 |
| `large_sample` | Large sample | 37 |
| `null` | Under the null | 34 |
| `weak` | Weak signal | 23 |
| `strong` | Strong signal | 22 |
| `misspecification` | Misspecified model | 19 |
| `high_noise` | High noise | 16 |
| `robustness` | Robustness | 16 |
| `sensitivity` | Sensitivity | 15 |
| `positive` | Effect present | 14 |
| `negative` | No effect | 12 |
| `low_noise` | Low noise | 6 |

## Using one

```python
import visualmetrics as vm

vm.lab("econometrics.heteroskedasticity", scenario="severe")
vm.concept("econometrics.heteroskedasticity").scenarios
```

In the GUI they are listed in the left panel, grouped by category, with the
canonical case open by default.
