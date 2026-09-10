"""Backpropagation is the chain rule evaluated in reverse."""

from __future__ import annotations

from .._kit import (
    EvidenceType,
    Level,
    ProofSpec,
    Reference,
    StepKind,
    VisualAction,
    assume,
    np,
    numeric_check,
    step,
)


def _check_gradient() -> tuple[float, str]:
    """Analytic backprop gradient against central finite differences."""
    rng = np.random.default_rng(5)
    n, d, h = 40, 4, 6
    X = rng.normal(size=(n, d))
    y = rng.normal(size=(n, 1))
    W1 = rng.normal(scale=0.5, size=(d, h))
    b1 = np.zeros((1, h))
    W2 = rng.normal(scale=0.5, size=(h, 1))
    b2 = np.zeros((1, 1))

    def forward(W1, b1, W2, b2):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        return z1, a1, z2, float(np.mean((z2 - y) ** 2))

    z1, a1, z2, loss = forward(W1, b1, W2, b2)
    # backward pass
    dz2 = 2.0 * (z2 - y) / n
    gW2 = a1.T @ dz2
    dz1 = (dz2 @ W2.T) * (1.0 - np.tanh(z1) ** 2)
    gW1 = X.T @ dz1

    eps = 1e-6
    worst = 0.0
    for W, g in ((W1, gW1), (W2, gW2)):
        for _ in range(12):
            i = rng.integers(W.shape[0])
            j = rng.integers(W.shape[1])
            orig = W[i, j]
            W[i, j] = orig + eps
            up = forward(W1, b1, W2, b2)[3]
            W[i, j] = orig - eps
            dn = forward(W1, b1, W2, b2)[3]
            W[i, j] = orig
            numeric = (up - dn) / (2 * eps)
            worst = max(worst, abs(numeric - g[i, j]))
    return worst, (
        f"largest gap between backprop and central differences over 24 sampled weights: {worst:.3e}"
    )


PROOF = ProofSpec(
    id="deep_learning.backprop.chain_rule",
    kind=EvidenceType.SYMBOLIC_DERIVATION,
    level=Level.ADVANCED,
    title="Backpropagation: the chain rule applied once, in reverse",
    claim=(
        "For a feedforward network, the gradient of the loss with respect to every weight "
        "is obtained from a single backward sweep of the recursion "
        "delta^(l) = (W^(l+1)' delta^(l+1)) * sigma'(z^(l)), at a total cost proportional "
        "to one forward pass."
    ),
    intuition=(
        "Each layer's error signal is the next layer's error signal pulled back through "
        "that layer's weights and then filtered by how responsive the layer's activation "
        "was. Because every layer reuses the signal computed by the layer above it, the "
        "whole gradient is assembled in one pass instead of being recomputed from scratch "
        "for every single weight."
    ),
    assumptions=(
        assume(
            "feedforward",
            "The computation is a directed acyclic graph of layers - no cycles.",
            if_violated=(
                "A recurrent network must first be unrolled through time; the same "
                "recursion then applies to the unrolled graph."
            ),
        ),
        assume(
            "differentiable",
            "The loss and every activation are differentiable at the current parameters.",
            essential=True,
            if_violated=(
                "ReLU is not differentiable at exactly zero. Practice substitutes a "
                "subgradient there, which is a defensible convention but not covered by "
                "this derivation."
            ),
        ),
        assume(
            "elementwise_activation",
            "Activations act element by element, so their Jacobian is diagonal.",
            essential=False,
            if_violated=(
                "For softmax or layer normalisation the Jacobian is a full matrix. The "
                "recursion still holds with the elementwise product replaced by a "
                "Jacobian-transpose product."
            ),
        ),
    ),
    steps=(
        step(
            "architecture",
            "Fix the forward computation.",
            equation=r"z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)}), "
            r"\quad a^{(0)} = x, \quad L = \ell(a^{(L)}, y)",
            why="Definition of a feedforward network.",
            kind=StepKind.SETUP,
            uses=("feedforward",),
            visual=VisualAction.TRACE_PATH,
        ),
        step(
            "define_delta",
            "Name the quantity worth caching: the sensitivity of the loss to a layer's "
            "pre-activation.",
            equation=r"\delta^{(l)} \equiv \frac{\partial L}{\partial z^{(l)}}",
            why=(
                "This is the whole trick. Everything a layer needs about the layers above "
                "it is summarised by this one vector, so nothing above has to be revisited."
            ),
            kind=StepKind.KEY_INSIGHT,
        ),
        step(
            "output_layer",
            "Start the recursion at the output layer.",
            equation=r"\delta^{(L)} = \nabla_{a^{(L)}}\ell \odot \sigma'(z^{(L)})",
            why="Chain rule through the final activation; the Jacobian of an elementwise map is diagonal.",
            kind=StepKind.CALCULUS,
            uses=("define_delta", "differentiable", "elementwise_activation"),
        ),
        step(
            "chain",
            "Push the sensitivity down one layer using the multivariate chain rule.",
            equation=r"\delta^{(l)}_j = \frac{\partial L}{\partial z^{(l)}_j} "
            r"= \sum_i \frac{\partial L}{\partial z^{(l+1)}_i}\,"
            r"\frac{\partial z^{(l+1)}_i}{\partial a^{(l)}_j}\,\frac{\partial a^{(l)}_j}{\partial z^{(l)}_j}",
            why=(
                "z^(l)_j influences the loss only through the units of layer l+1, so the "
                "total derivative sums over exactly those paths."
            ),
            kind=StepKind.CALCULUS,
            uses=("architecture", "define_delta"),
            visual=VisualAction.UNFOLD_CHAIN,
        ),
        step(
            "partials",
            "Read the two inner partial derivatives off the forward equations.",
            equation=r"\frac{\partial z^{(l+1)}_i}{\partial a^{(l)}_j} = W^{(l+1)}_{ij}, \qquad "
            r"\frac{\partial a^{(l)}_j}{\partial z^{(l)}_j} = \sigma'(z^{(l)}_j)",
            why="Differentiating the affine map and the elementwise activation.",
            kind=StepKind.SUBSTITUTION,
            uses=("architecture", "elementwise_activation"),
        ),
        step(
            "recursion",
            "Substituting gives the backward recursion in matrix form.",
            equation=r"\delta^{(l)} = \big(W^{(l+1)\prime}\delta^{(l+1)}\big) \odot \sigma'(z^{(l)})",
            why="The sum over i is precisely a matrix-vector product with the transposed weights.",
            kind=StepKind.KEY_INSIGHT,
            uses=("chain", "partials"),
            visual=VisualAction.UNFOLD_CHAIN,
        ),
        step(
            "weight_gradients",
            "Every parameter gradient is then a local outer product.",
            equation=r"\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)}a^{(l-1)\prime}, \qquad "
            r"\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}",
            why=(
                "z^(l) = W^(l)a^(l-1) + b^(l) is linear in the parameters, so its "
                "derivative with respect to W^(l)_{jk} is a^(l-1)_k."
            ),
            kind=StepKind.CONCLUSION,
            uses=("recursion", "architecture"),
            visual=VisualAction.HIGHLIGHT_TERM,
        ),
        step(
            "cost",
            "Count the work: one forward pass, one backward pass, all gradients.",
            equation=r"O(P) \ \text{ versus } \ O(P^2) \ \text{ for finite differences, } P = \#\text{parameters}",
            why=(
                "The backward pass touches each weight a constant number of times. "
                "Central differences would need two forward passes per parameter, each "
                "costing O(P) - and would still only be an approximation."
            ),
            kind=StepKind.CONCLUSION,
            uses=("weight_gradients",),
        ),
        step(
            "vanishing",
            "The recursion also explains vanishing and exploding gradients.",
            equation=r"\delta^{(l)} = \Big(\prod_{m=l+1}^{L} D^{(m-1)}W^{(m)\prime}\Big)\delta^{(L)}, "
            r"\quad D^{(m)} = \operatorname{diag}(\sigma'(z^{(m)}))",
            why=(
                "Unrolling the recursion turns it into a product of matrices. Repeated "
                "multiplication by factors whose typical magnitude is below one shrinks "
                "the signal geometrically with depth; above one, it grows geometrically. "
                "Saturating activations, where sigma' is near zero, push the product "
                "toward the first case."
            ),
            kind=StepKind.CONCLUSION,
            uses=("recursion",),
            visual=VisualAction.HIGHLIGHT_TERM,
        ),
    ),
    conclusion=(
        "Backpropagation is not a separate learning principle. It is the chain rule with "
        "the intermediate sensitivities cached and reused, which is why it produces exact "
        "gradients - up to floating-point error - for the price of roughly one extra "
        "forward pass."
    ),
    limitations=(
        "This derives the gradient, nothing more. It gives no guarantee that gradient "
        "descent converges, that it reaches a global minimum, or that the resulting "
        "network generalises - the loss surface of a deep network is non-convex and this "
        "argument is silent about its shape. It assumes exact arithmetic: in float32 the "
        "product of many small factors underflows, which is a numerical failure on top of "
        "the mathematical vanishing described above. The differentiability assumption is "
        "violated at the ReLU kink. Nothing here explains why depth helps."
    ),
    prerequisites=(),
    concept_ids=("deep_learning.backpropagation", "ml.gradient_descent"),
    references=(
        Reference(
            "Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986). Learning "
            "representations by back-propagating errors. Nature 323, 533-536.",
            kind="paper",
            doi="10.1038/323533a0",
        ),
        Reference(
            "Goodfellow, I., Bengio, Y. and Courville, A. (2016). Deep Learning, ch. 6.5.",
            kind="book",
        ),
    ),
    checks=(
        numeric_check(
            "matches_finite_differences",
            "Compare the analytic backprop gradient with central finite differences on a "
            "small tanh network.",
            _check_gradient,
            tol=1e-8,
        ),
    ),
)
