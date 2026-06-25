# Neural-Network Support in Rumoca — Design & Implementation Plan

**Status:** native training path **implemented** (2026-06-25); inference/codegen partly done. Direction: **native-first** (the JAX/diffrax training path is deprioritized — see note below).
**Original date:** 2026-06-16 · **Revised:** 2026-06-25 (rebased onto rumoca v0.9.9)

> **Implementation status (2026-06-25).** Landed on branch `nn-support-reverse-ad`:
> 1. **Native MatMul codegen** — `jax-solve`/`casadi-solve` now emit real `jnp.matmul`/`ca.mtimes`
>    (not scalar fallback); `[capabilities.tensor] matmul = "native"` declared.
> 2. **`__rumoca(trainable=true)` annotation** → `trainable: bool` on the DAE parameter `Variable`.
> 3. **Native trajectory adjoint** — `SolveRuntime::trajectory_euler_adjoint` (explicit-Euler
>    discrete adjoint), reusing the per-step reverse VJP; matches forward-sensitivity to ~1e-14.
> 4. **Native Adam + MSE training loop** — `SolveRuntime::train_trajectory_mse`; recovers known
>    params to machine-zero loss.
> 5. **End-to-end NN training** — a 1→2→1 tanh net embedded in `der(x)`, all 7 weights trainable by
>    annotation, trained natively (loss 3.4e-2 → 6.4e-7), **no Python**.
>
> **Known limits / next:** the trajectory adjoint is **pure-ODE only** — a dense layer written as
> `matmul → hidden-vector algebraic` introduces solver algebraics it rejects, so NN-in-RHS models
> must currently be inlined to a pure-ODE form. Lifting this (algebraic/DAE trajectory adjoint) and
> RK45/variable-step adjoint are the next native lifts. The `trainable` flag is not yet plumbed into
> the eval-solve runtime (trainable slots are resolved at the test/compile layer via
> `param_labels`/`param_slots`). **The JAX/diffrax codegen-training path was intentionally dropped
> per the native-only decision** — the matmul codegen above still benefits the JAX/CasADi backends.

> **What changed in this revision.** The plan was first written against a tree that had
> **forward-mode AD only**. The codebase has since advanced (now v0.9.9, post-PR #267):
>
> 1. **Native reverse-mode AD landed.** There is now a scalar reverse VJP over the Solve IR
>    (`crates/rumoca-eval-solve/src/reverse.rs`) and a **steady-state adjoint** objective gradient
>    via matrix-free GMRES (`crates/rumoca-eval-solve/src/jacobian.rs`), exposed to Python as
>    `Model.objective_gradient(objective, state, t, mode="adjoint")`
>    (`crates/rumoca-bind-python/src/gradient.rs`). This **partially unblocks native training**
>    (see §4.5). Caveat: it is **steady-state only** and **native-Rust only** (not emitted by the
>    JAX/CasADi codegen backends).
> 2. **Select-chain bloat is fixed** for runtime weight indexing (PR #267 added `LoadIndexedP` /
>    `LoadIndexedSeed`). This removes the headline scalar/C/FMU blocker.
> 3. **Two new solve-IR codegen targets exist:** `jax-solve` and `casadi-solve`, which render the
>    RHS directly from the Solve IR rather than the DAE IR. **These are the new home for NN
>    codegen** and replace most of the old `jax.py.jinja` work.
> 4. Several Phase-0 "unknown/verify" frontend gaps are now **confirmed working** (vectorization,
>    function redeclaration). The remaining real gap is **MatMul lowering** — see §3.

---

## 1. Goal & the two value propositions

"NN support" can mean two distinct things. Rumoca should target both — they have very different
cost and very different payoff.

1. **Inference / surrogates.** A pre-trained NN runs *inside* a model — as a fast surrogate, or
   to add data-driven accuracy. Training happens elsewhere; the model only evaluates the network.
2. **Differentiable training (neural ODEs).** Small NNs are embedded in the *right-hand side of
   the ODEs* (e.g. a neural spring/damper), and the weights — plus selected physical parameters —
   are trained **through the simulation** with a differentiable ODE solver.

**Why rumoca is well-positioned:** it already emits **JAX (with diffrax)** and **CasADi**, both of
which give reverse-mode AD through the ODE solve. So rumoca can produce a *native, trainable* model
directly from Modelica — without exporting an opaque FMU and wrapping it in a separate training
host. General-purpose Modelica tools don't do this. **That is the differentiator and should drive
the architecture.** As of v0.9.9 rumoca *also* has its own **native reverse-mode adjoint** (Rust,
no Python) — today limited to steady-state objectives, but a clear path to in-process training of
small embedded NNs without any external autograd framework (§4.5).

"Native-first" = build the generic frontend + tensor-codegen capabilities so *all* backends
benefit, then layer training on top — rather than special-casing one backend.

---

## 2. Layer representation (our design)

A dense layer reduces to **matmul + bias add + element-wise activation**:

```modelica
block Dense
  parameter Real weights[:,:];
  parameter Real bias[:];
  replaceable function f = Activation;   // activation redeclared per layer
equation
  y = f(weights * u + bias);             // f auto-vectorized over the result vector
end Dense;
```

Activations are plain Modelica functions (`relu` = `max(0,u)`, `tanh`, `sigmoid`, `softplus`,
`identity`, …). Weights/biases are `parameter Real[...]` arrays. Layers wire together with
`connect`. Whole networks are ordinary Modelica blocks.

**Design decision:** lower NNs through the *existing* pipeline as ordinary Modelica — do **not**
add a heavyweight `NeuralLayer` IR node in phase 1. A layer is matmul (existing tensor op) +
elementwise activation (existing scalar ops + vectorization). Benefits: every backend inherits
support, no new IR surface to maintain, and any model written this way "just works." Optional
later sugar (phase 4): a pattern recognizer that spots `activation(W*u + b)` so the torch/keras
backends can emit `nn.Linear` for readability/perf — cosmetic, not required.

This decision is **validated**: the two frontend primitives the design leans on — auto-vectorizing
a scalar activation over an array, and redeclaring the activation per layer — are both confirmed
working in the current tree (§3). The one place "ordinary lowering" currently costs us is MatMul,
which the solve-IR codegen flattens to scalar element ops rather than a real matmul (§3, Phase 1).

---

## 3. Current rumoca capabilities vs. gaps (grounded findings, v0.9.9)

Pipeline: AST → Flat → DAE → Solve IR → template codegen (minijinja). Backends include
`jax` (diffrax), `jax-solve`, `casadi-mx`/`casadi-sx`, `casadi-solve`, `onnx`, `sympy`, `symforce`,
`julia-mtk`, `fmi2`/`fmi3`, `c-solve`, `embedded-c`, `rust-solve`, `cuda`, `wgsl`, `mlir`,
cranelift JIT. Per-target `target.toml` declares capabilities (incl. `tensor`, `external_tables`,
`forward_ad`, `reverse_ad`).

| Capability needed by an NN model | State today (v0.9.9) | Notes / file |
|---|---|---|
| Auto element-wise vectorization of scalar fn over array `f(vec)` | **✓ works** | `crates/rumoca-phase-flatten/src/functions/call_args.rs:158-200` (`validate_vectorized_dimensions`, MLS §12.4.6). Gates dense layers; now confirmed. |
| `replaceable function f` + `redeclare function f = tanh` | **✓ works** | Resolved in instantiate: `crates/rumoca-phase-instantiate/src/inheritance.rs:35,209,232`; error surface in `.../errors.rs:119-160`. Per-layer activation redeclare is fine. |
| `weights * u` (matrix·vector / matmul) | **✗ gap — scalar fallback** | MatMul is a first-class Solve-IR `ComputeNode` (`crates/rumoca-ir-solve/src/lib.rs:21-43`), **but** the new `jax-solve`/`casadi-solve` templates route it through `push_multi_output_tensor_fallback_program` → expands to *m·n scalar `*` ops*, not `jnp.matmul`/`@` or `ca.mtimes` (`crates/rumoca-phase-codegen/src/codegen/render_solve/template_partition.rs:704-714`; `templates/jax-solve/jax_solve.py.jinja:57`). **This is the #1 inference gap.** |
| Literal matrix/vector parameters `Real W[m,n]={...}` | mixed: old `jax` ✗, solve-IR path uses runtime `p` | Old DAE template still casts params with `float(...)` → arrays collapse to `0.0` (`templates/jax/jax.py.jinja:106,114,...`). The **new `jax-solve`/`casadi-solve` path sidesteps this** by feeding weights through the runtime parameter vector `p` rather than literal initializers — prefer that path. |
| Runtime-indexed weight loads (no select-chain bloat) | **✓ fixed (PR #267)** | `LoadIndexedP { base, count, index }` / `LoadIndexedSeed` (`crates/rumoca-ir-solve/src/linear_op.rs:175-196`; render `crates/rumoca-phase-codegen/src/codegen/render_solve.rs:595-613`). PR #267: quadrotor 88,002→402 selects, generated C 143 MB→1.8 MB. **Scalar/C/FMU weight indexing no longer explodes.** |
| `connect` of Real vector signals between blocks | Likely ✓ | Standard connection handling. |
| Activation functions as user Modelica functions | Preserved as backend `def`s, not inlined | `render_expr.rs` user-fn lowering; JAX/CasADi trace through the emitted `def`. So **no NN-specific builtins are strictly required** if user-fn lowering + vectorization work (they do). |
| Activation builtins (sigmoid/relu/gelu/softmax) | Missing as builtins; `tanh,sinh,cosh,exp,log,max,min,abs,sqrt,…` present | `crates/rumoca-phase-codegen/src/codegen/render_expr.rs:577-678`. `relu`=`max(0,u)`, `sigmoid`=`1/(1+exp(-u))` already expressible. Only needed as ergonomic/interop sugar. |
| External-table file loading for weights | C/FMI only; not needed for JAX/CasADi | **Weights are just parameters** → for JAX/CasADi/torch "loading weights" = supplying the runtime `p` vector; no compiler change needed. |

**Key takeaway (revised):** the Phase-0 unknowns have largely resolved in our favour —
vectorization, function redeclaration, and weight-indexing bloat are all handled. The make-or-break
remaining frontend/codegen work for *native inference* is now narrow and concrete:
**lower MatMul to a real matmul in the `jax-solve`/`casadi-solve` templates** (and feed weights via
the runtime `p` vector, not literal `float()` params).

---

## 4. Design

### 4.1 Representation
Keep NNs as ordinary Modelica composed from existing primitives (see §2). No `NeuralLayer` IR
node in phase 1. Target the **solve-IR codegen path** (`jax-solve`, `casadi-solve`) — not the older
DAE-IR `jax`/`casadi-mx` templates — because that path already handles runtime parameter vectors
and the indexed-load lowering, and it's where MatMul rendering needs to be fixed once.

### 4.2 Trainable-parameter class (the one genuinely new concept)

Training needs to distinguish **learnable** params (weights/bias, and opted-in physical params)
from **fixed** physical params. Proposal:

- Mark via vendor annotation, e.g. `parameter Real W[...] annotation(__rumoca(trainable=true));`
  (and/or auto-tag params inside components extending an NN base block).
- Carry a `trainable: bool` flag on DAE/Solve-IR parameters.
- Codegen splits the param vector into `theta` (trainable) and `p` (fixed):
  - **JAX (`jax-solve`):** `theta` becomes a separate pytree leaf; model is `f(theta, p, x, t)` so
    `jax.grad` / diffrax adjoint flows through `theta`.
  - **Torch:** `theta` entries become `nn.Parameter(requires_grad=True)`; `p` are buffers.
  - **Native Rust:** `theta` are the columns the adjoint/sensitivity machinery differentiates
    against (the parameter cotangent slots in `reverse_*_vjp`, §4.5).
- This also enables fitting selected *physical* parameters alongside the weights in one optimizer.

### 4.3 Weight import strategy

- **Phase 1 — runtime parameter vector (preferred).** Weights are parameters → for
  JAX/CasADi/torch the caller supplies the `p`/`theta` vector at runtime from a `.npz`/`.json`/
  `.mat`. The new solve-IR path already routes parameters this way, so this needs **no compiler
  change** beyond a documented param layout + a tiny Python loader helper. (Literal `={...}`
  initializers also work for small nets on the solve-IR path, but the old DAE-IR `float()` bug
  means *avoid the older DAE-IR `jax` target* for matrix params.)
- **Phase 2 — ONNX importer** (separate preprocessing tool). Reads `.onnx`, emits either (a) a
  Modelica network block, or (b) a weights file + thin Modelica loader. ONNX is the ML-ecosystem
  interchange format.

### 4.4 Backend matrix (target end-state)

| Backend | Inference | Training (AD through solve) | Priority |
|---|---|---|---|
| JAX (`jax-solve` + diffrax) | ✓ (after MatMul fix) | ✓ (jax.grad + diffrax adjoint) | **highest** — closest to done |
| CasADi (`casadi-solve`, mx/sx) | ✓ (after MatMul fix) | ✓ (symbolic AD; opt/NLP) | high |
| Torch (+torchdiffeq) | ✓ | ✓ (autograd) | high — new backend |
| **Native Rust (Solve-IR evaluator)** | ✓ | **steady-state ✓ today; trajectory = future** (§4.5) | high — unique differentiator |
| C / FMI / embedded | ✓ (indexed-loads fix landed) | inference/deploy only | medium |
| ONNX / others | export | n/a | low |

### 4.5 Native train + run (no Python) — feasibility *(substantially revised)*

**Running natively: yes.** rumoca has a full native Rust sim path (`rumoca-solver-diffsol`
stiff/implicit BDF, `rumoca-solver-rk45`, Solve-IR evaluator, cranelift JIT). An embedded NN runs
there with no Python — and with the PR #267 indexed-load fix, large-NN weight indexing no longer
select-chain-bloats.

**Training natively: now partially supported (this is the big change).** The native AD is no longer
forward-only. The tree now has:

- **Scalar reverse VJP** over the Solve IR — `reverse_scalar_block_vjp` and the runtime wrapper
  `SolveRuntime::reverse_state_derivative_vjp` compute `(∂der/∂[solver_y | p])ᵀ · λ` exactly
  (symbolic, not finite-difference), with reusable scratch buffers for allocation-free loops
  (`crates/rumoca-eval-solve/src/reverse.rs`, `crates/rumoca-eval-solve/src/runtime/sensitivity.rs`).
  Verified by the dot-product identity `λᵀ(Jv) = (Jᵀλ)ᵀv` (`crates/rumoca/tests/reverse_vjp_test.rs`).
- **Steady-state adjoint objective gradient** — `steady_state_adjoint_objective_gradient` settles
  the algebraic system, then solves the transposed residual system `(∂R/∂y)ᵀ λ = e_objective` via
  **matrix-free GMRES** and returns `dJ/dp = −(∂R/∂p)ᵀ λ`. Handles states *and* nonlinear
  algebraics via the implicit-function theorem (`crates/rumoca-eval-solve/src/jacobian.rs`;
  `crates/rumoca/tests/steady_adjoint_test.rs`).
- **Python surface** — `Model.objective_gradient(objective, state, t, mode="adjoint")` (also
  `"forward"`) returns a `GradientResult` with `to_dict()/to_numpy()/to_series()`
  (`crates/rumoca-bind-python/src/gradient.rs`).

**What this unblocks:** in-process, exact gradients of a **steady-state** objective w.r.t. all
parameters (including NN weights), in Rust, with no autograd framework. Combined with a small Rust
optimizer (Adam / Gauss-Newton / LM) this gives **fully native training of steady-state-objective
NN/parameter-fitting problems** — no Python in the loop.

**What's still missing for the headline neural-ODE case:**

1. **Through-time (trajectory) adjoint is not implemented.** The reverse path covers a single RHS
   VJP and the *steady-state* adjoint; it does **not** backprop through the time integration of an
   ODE over a horizon. `reverse_state_derivative_vjp` explicitly rejects models with solver
   algebraics, and there is no adjoint-ODE integrator. Training a neural-ODE against *trajectory*
   data natively requires building the continuous/discrete adjoint over the solver — a bounded but
   real undertaking (reuse the existing VJP as the per-step primitive; integrate the adjoint
   backward in time). **This is the main remaining native-training gap.**
2. **No optimizer / loss / training loop in-tree.** The gradient is a pure function; there is no
   Adam/GN/LM, loss library, batching, or training driver yet. Adding a small Rust optimizer is
   low-risk and reuses the gradient primitives above.
3. **Native reverse AD is not wired into the codegen backends.** It lives in the Rust evaluator
   only; `jax-solve`/`casadi-solve` still rely on their own framework autograd. That's fine — those
   backends get AD for free — but the native and codegen training paths are separate.

**Two paths to full native neural-ODE training, in order:**

1. **Steady-state / forward-sensitivity training (available now).** For steady-state objectives use
   the new adjoint directly. For dynamic objectives on *small* NNs, the existing forward
   sensitivities (∂state/∂θ from the stiff solver) + a Rust optimizer already train in-process; cost
   ~O(#trainable params), acceptable for the hundreds-of-params neural-ODE regime.
2. **Trajectory adjoint through the solve (scalable).** Build the adjoint-ODE integrator so reverse
   cost is independent of #params. Needed when params ≫ outputs. Defer until #1 proves too slow.

For comparison, the `jax-solve`/`casadi-solve`/torch backends still give *dynamic* training soonest
(free autograd through diffrax/torchdiffeq/CasADi); that path only "leaves rumoca" as a thin
generated Python script. The native path is the answer when the requirement is genuinely
end-to-end Rust — and it is now real for steady-state, with trajectory adjoint the one remaining
lift.

---

## 5. Phased roadmap *(re-scoped to v0.9.9 reality)*

### Phase 0 — Validation — **mostly complete**
The original Phase-0 unknowns are resolved (§3): vectorization ✓, function redeclaration ✓,
indexed-load bloat ✓ fixed. Remaining validation, narrowed to one concrete check: run a small dense
NN through **`jax-solve`** and **`casadi-solve`** and confirm exactly how `weights * u` is emitted
(expected: scalar fallback today). Capture the generated RHS as the regression baseline for Phase 1.

### Phase 1 — Native inference (the MatMul fix)
Goal: a dense feed-forward NN compiles and simulates correctly **and efficiently** on `jax-solve` +
`casadi-solve`.
- **Lower MatMul to a real matmul** in the solve-IR templates instead of the scalar tensor
  fallback: emit `jnp.matmul`/`@` (JAX) and `ca.mtimes` (CasADi) for the `MatMul` `ComputeNode`
  (`render_solve/template_partition.rs:704-714`; `jax-solve`/`casadi-solve` `.jinja`). Declare the
  resulting `[capabilities.tensor]` (`matmul: "native"`) in the two new `target.toml`s, which
  currently declare none.
- Feed weights via the runtime parameter vector `p` (preferred) and/or via solve-IR literal params;
  **do not** use the older DAE-IR `jax` target for matrix params (its `float()` cast collapses
  arrays).
- (Optional) add `sigmoid`/`relu`/`gelu`/`softmax` builtins as interop sugar in `render_expr.rs`.
- **Milestone:** a known NN surrogate's prediction matches a reference within tolerance, with the
  generated RHS using a real matmul (not m·n scalar `*` ops).

### Phase 2 — Trainable parameters + Torch backend
- Add `trainable` param flag + `__rumoca(trainable=...)` annotation; thread it through to Solve-IR
  params.
- `jax-solve` codegen: split `theta`/`p`; expose `model(theta, p, x, t)`.
- New `torch` template (mirror `jax-solve`): `nn.Module`, `nn.Parameter` for `theta`, `torchdiffeq`
  for the ODE path.
- Native: expose a small Rust optimizer (Adam first) driving `objective_gradient`/`reverse_*_vjp`
  for steady-state fits.
- **Milestone:** generated JAX/Torch model trains a toy embedded NN to fit data (grad flows); native
  Rust fits a steady-state objective end-to-end with no Python.

### Phase 3 — Neural-ODE example + weight I/O
- Author a hybrid demo: a small NN embedded in an ODE RHS (e.g. neural friction/damper term) +
  a training script (optimizer, mini-batching, optional physically-motivated regularizers) on
  `jax-solve`/torch.
- Runtime weight injection helper (`.npz`/`.json`) + documented param layout.
- **Milestone:** learn a nonlinearity from trajectory data end-to-end (codegen path).

### Phase 4 — Native trajectory adjoint + scale & polish
- **Native through-time adjoint:** build the adjoint-ODE integrator over the solver so neural-ODEs
  train fully in Rust against trajectory data, reverse cost independent of #params (reuses the
  Phase-0/§4.5 per-step VJP as the primitive). Lift `reverse_state_derivative_vjp`'s pure-ODE
  restriction to cover the algebraic/DAE case along the trajectory.
- ONNX importer tool.
- Optional `nn.Linear` pattern sugar for torch/keras.
- LSTM/recurrent: requires clocked/synchronous equation support (`sample`/`hold`/`previous`) —
  larger separate lift; feed-forward + neural-ODE don't need it.

---

## 6. Risks / open questions *(revised)*

- **MatMul scalar fallback (now the #1 issue).** The new `jax-solve`/`casadi-solve` templates expand
  `MatMul` to m·n scalar `*` ops rather than a native matmul. Functionally correct but defeats the
  performance rationale for choosing JAX/CasADi for dense layers. Phase 1 must fix the template
  rendering, not the frontend.
- **Two JAX targets — pick the right one.** Use `jax-solve` (solve-IR, runtime `p`, indexed loads),
  **not** the older `jax` (DAE-IR, `float()`-casts array params to `0.0`). Document this so models
  don't silently get zeroed weights.
- **Native trajectory adjoint missing.** Native reverse AD covers single-RHS VJP + steady-state
  adjoint only. Dynamic neural-ODE training natively needs the through-time adjoint (Phase 4);
  until then, dynamic native training falls back to forward sensitivities (fine for tiny NNs) or to
  the JAX/torch/CasADi codegen path.
- **Resolved (no longer risks):** function redeclaration ✓, scalar-fn-over-array vectorization ✓,
  and select-chain bloat for weight indexing ✓ (PR #267). These were the biggest Phase-0 unknowns.
- Reverse-mode AD for the *codegen* backends still comes from diffrax/torchdiffeq/casadi — rumoca
  only needs to emit clean differentiable code (and a real matmul) there; its own native reverse AD
  is a separate, additive path, not on the JAX/torch training critical path.
