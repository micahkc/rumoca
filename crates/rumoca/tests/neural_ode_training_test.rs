//! Capstone: train a neural network embedded in an ODE right-hand-side, fully
//! native (no Python), with reverse-mode discrete-adjoint AD.
//!
//! A tiny dense net `1 -> 2 -> 1` with `tanh` activation parameterizes the
//! dynamics `der(x) = -NN(x)`. The hidden/output weights and biases are the
//! trainable matrix/vector parameters `W1[2,1]`, `b1[2]`, `W2[1,2]`, `b2`, each
//! flagged learnable purely by the Modelica annotation
//! `annotation(__rumoca(trainable=true))`. We
//!
//! 1. compile the model to a [`SolveRuntime`] (this exercises trainable
//!    matrix/vector parameters + `tanh` activations, all on `nn-support-reverse-ad`);
//! 2. generate synthetic target data by Euler-integrating a known target
//!    dynamics `der(x) = -k*x*(1 + x^2)`;
//! 3. derive the trainable slots *from the annotation* (Part 1), not by
//!    hard-coding indices;
//! 4. train the NN weights end-to-end with [`SolveRuntime::train_trajectory_mse`]
//!    and assert the loss collapses and the trained trajectory tracks the target.
//!
//! The dense layer is written with explicit element indexing of the trainable
//! weight matrices rather than a `W * {x}` array temporary: the native
//! trajectory adjoint reverses only *pure-ODE* models (no solver algebraics —
//! the algebraic-projection adjoint is "Track B" and not yet implemented), and a
//! `W1 * {x}` matmul materializes the hidden vector `h` as an algebraic. The
//! scalarized reverse AD differentiates the indexed form identically — every
//! weight is still a genuine trainable `parameter Real` slot driving two tanh
//! hidden units summed into the output.

use rumoca::Compiler;
use rumoca_eval_solve::{AlgebraicSettle, EulerGrid, SolveRuntime, TrainConfig};
use rumoca_ir_dae::Dae;
use rumoca_sim::SimOptions;

/// Neural-ODE model: a dense `1 -> 2 -> 1` net (two tanh hidden units, weighted
/// output + bias) drives `der(x) = -NN(x)`. The trainable matrix/vector
/// parameters `W1`, `b1`, `W2`, `b2` are the network weights, indexed
/// element-wise so the right-hand side stays a pure ODE (no algebraic temporary).
const NN_SOURCE: &str = r#"
model NeuralOde
  parameter Real W1[2,1] = {{0.5},{-0.3}} annotation(__rumoca(trainable=true));
  parameter Real b1[2]   = {0.1, -0.2}    annotation(__rumoca(trainable=true));
  parameter Real W2[1,2] = {{0.2, -0.1}}  annotation(__rumoca(trainable=true));
  parameter Real b2      = 0.0            annotation(__rumoca(trainable=true));
  Real x(start = 1.0);
equation
  der(x) = -( W2[1,1]*tanh(W1[1,1]*x + b1[1])
            + W2[1,2]*tanh(W1[2,1]*x + b1[2])
            + b2 );
end NeuralOde;
"#;

const T0: f64 = 0.0;
const H: f64 = 1.0e-3;
const STEPS: usize = 150;

fn settle() -> AlgebraicSettle {
    AlgebraicSettle {
        tol: 1.0e-12,
        max_iters: 64,
    }
}

fn build_runtime(source: &str, model: &str) -> (SolveRuntime, rumoca_ir_solve::SolveModel, Dae) {
    let result = Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|e| panic!("{model} should compile: {e:?}"));
    let solve_model = rumoca_sim::lower_dae_for_simulation(&result.dae, &SimOptions::default())
        .expect("lowering should succeed");
    let runtime = SolveRuntime::new(&solve_model).expect("runtime should build");
    (runtime, solve_model, result.dae)
}

/// Part 1 (test/compile layer): derive the trainable parameter slots straight
/// from the `trainable` annotation. The flag lives on the DAE parameter
/// `Variable` (it is intentionally *not* plumbed into solve-IR / `SolveRuntime`,
/// since the runtime treats every `p_scalars` slot identically), so we read the
/// trainable parameter *names* from the DAE and map them to slots via the
/// parameter-Jacobian report's `param_labels`/`param_slots` — exactly the same
/// name space the runtime exposes.
///
/// Array weights occupy several scalar P-slots; their `param_labels` are the
/// array base name plus subscripted element names (e.g. trainable `W1` maps to
/// labels `"W1"` and `"W1[2,1]"`). So a trainable DAE parameter `name` claims
/// every slot whose label is exactly `name` or begins with `name[`.
fn trainable_slots(runtime: &SolveRuntime, dae: &Dae, params: &[f64]) -> Vec<usize> {
    let report = runtime.eval_parameter_jacobian(0.0, &[1.0], params, settle());
    assert!(report.error.is_none(), "param jacobian: {:?}", report.error);

    let mut slots: Vec<usize> = Vec::new();
    for (name, _) in dae.variables.parameters.iter().filter(|(_, v)| v.trainable) {
        let base = name.as_str();
        let subscript_prefix = format!("{base}[");
        for (label, &slot) in report.param_labels.iter().zip(report.param_slots.iter()) {
            if label == base || label.starts_with(&subscript_prefix) {
                slots.push(slot);
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Euler-integrate a scalar-`x` trajectory under a closed-form target dynamics
/// `der(x) = -k*x*(1 + x^2)`, producing the synthetic observed data the NN must
/// learn to reproduce.
fn target_trajectory(x0: f64, k: f64) -> Vec<Vec<f64>> {
    let mut x = x0;
    let mut states = Vec::with_capacity(STEPS + 1);
    states.push(vec![x]);
    for _ in 0..STEPS {
        let der = -k * x * (1.0 + x * x);
        x += H * der;
        states.push(vec![x]);
    }
    states
}

#[test]
fn neural_ode_trains_dense_layer_natively() {
    let x0_val = 1.0_f64;
    let (runtime, solve_model, dae) = build_runtime(NN_SOURCE, "NeuralOde");

    // Pure ODE: the single state `x`, no solver algebraics (required by the
    // native trajectory adjoint).
    assert_eq!(runtime.state_count, 1, "single ODE state `x`");
    assert_eq!(runtime.solver_count, 1, "pure ODE: solver_y == states");
    let x0 = vec![x0_val];

    // --- Part 1: trainable slots straight from the annotation. ---
    let slots = trainable_slots(&runtime, &dae, &solve_model.parameters);
    // 2 (W1) + 2 (b1) + 2 (W2) + 1 (b2) = 7 trainable scalar weights.
    assert_eq!(
        slots.len(),
        7,
        "expected 7 trainable scalar weights from the annotation, got {slots:?}"
    );

    // --- Synthetic target trajectory from a known nonlinear dynamics. ---
    let observed = target_trajectory(x0_val, 1.5);
    let targets: Vec<&[f64]> = observed.iter().map(|s| s.as_slice()).collect();

    // --- Start from the model's small initial weights and train. ---
    let mut params = solve_model.parameters.clone();
    let cfg = TrainConfig {
        grid: EulerGrid {
            t0: T0,
            h: H,
            steps: STEPS,
        },
        settle: settle(),
        epochs: 8000,
        lr: 1.0e-2,
    };

    let report = runtime
        .train_trajectory_mse(&x0, &mut params, &slots, &targets, &cfg)
        .expect("native NN training should run");

    // --- Loss curve + trajectory tracking error. ---
    let n = report.losses.len();
    eprintln!("trainable slots = {slots:?}");
    eprintln!("epochs = {n}");
    for &i in &[0usize, n / 8, n / 4, n / 2, 3 * n / 4, n - 1] {
        eprintln!("  epoch {i:>5}: loss = {:.6e}", report.losses[i]);
    }
    eprintln!("initial_loss = {:.6e}", report.initial_loss);
    eprintln!("final_loss   = {:.6e}", report.final_loss);

    // Re-integrate with the trained weights and measure max per-sample error.
    let trained = euler_with_runtime(&runtime, &x0, &params);
    let max_err = trained
        .iter()
        .zip(observed.iter())
        .map(|(p, t)| (p[0] - t[0]).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("max per-sample trajectory error = {max_err:.6e}");

    // --- Assertions (not loosened): loss collapses and trajectory tracks. ---
    assert!(
        report.final_loss < report.initial_loss / 50.0,
        "loss should drop by >50x: initial = {:.6e}, final = {:.6e}",
        report.initial_loss,
        report.final_loss
    );
    assert!(
        report.final_loss < 1.0e-6,
        "final loss should be < 1e-6, got {:.6e}",
        report.final_loss
    );
    assert!(
        max_err < 1.0e-2,
        "trained trajectory should track target within 1e-2, got {max_err:.6e}"
    );
}

/// Euler-integrate the compiled runtime's own dynamics with `params`, returning
/// stored states `x0 … x_N` for trajectory-tracking comparison.
fn euler_with_runtime(runtime: &SolveRuntime, x0: &[f64], params: &[f64]) -> Vec<Vec<f64>> {
    let settle = settle();
    let mut state = x0.to_vec();
    let mut der = vec![0.0; runtime.state_count];
    let mut states = Vec::with_capacity(STEPS + 1);
    states.push(state.clone());
    for n in 0..STEPS {
        let t_n = T0 + H * n as f64;
        runtime
            .eval_state_derivatives_into(
                t_n,
                &state,
                params,
                settle.tol,
                settle.max_iters,
                &mut der,
            )
            .expect("primal derivative");
        for i in 0..runtime.state_count {
            state[i] += H * der[i];
        }
        states.push(state.clone());
    }
    states
}
