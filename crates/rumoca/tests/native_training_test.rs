//! Fully-native parameter training: tune model parameters from trajectory data
//! using the native reverse-mode (discrete-adjoint) gradient and Adam — no
//! Python. The mechanism is identical whether the trained slots are NN weights
//! or physical constants; here we run a parameter-recovery experiment.
//!
//! 1. Generate synthetic "observed" data by Euler-integrating with ground-truth
//!    `a* = 0.7, b* = 0.3`.
//! 2. Start the optimizer from a perturbed guess (`a = 0.3, b = 0.8`) and fit
//!    the `a, b` slots (resolved via the parameter-Jacobian report, not
//!    hard-coded).
//! 3. Assert the loss collapses and the recovered parameters return to ground
//!    truth.

use rumoca::Compiler;
use rumoca_eval_solve::{AlgebraicSettle, EulerGrid, SolveRuntime, TrainConfig};
use rumoca_sim::SimOptions;

// Pure ODE, strictly smooth and bounded (the proven `TrajModel`).
const SOURCE: &str = r#"
model TrajModel
  parameter Real a = 0.7;
  parameter Real b = 0.3;
  Real x(start = 0.5);
  Real y(start = -0.2);
equation
  der(x) = sin(a*x) + b*y*y;
  der(y) = tanh(a*y) - b*x;
end TrajModel;
"#;

const T0: f64 = 0.0;
const H: f64 = 1.0e-3;
const STEPS: usize = 200;

fn build_runtime() -> (SolveRuntime, rumoca_ir_solve::SolveModel) {
    let result = Compiler::new()
        .model("TrajModel")
        .compile_str(SOURCE, "TrajModel.mo")
        .expect("TrajModel should compile");
    let solve_model = rumoca_sim::lower_dae_for_simulation(&result.dae, &SimOptions::default())
        .expect("lowering should succeed");
    let runtime = SolveRuntime::new(&solve_model).expect("runtime should build");
    (runtime, solve_model)
}

fn settle() -> AlgebraicSettle {
    AlgebraicSettle {
        tol: 1.0e-12,
        max_iters: 64,
    }
}

/// Euler-integrate the trajectory with the given parameters, returning the
/// stored states `x₀ … x_N` (the synthetic "observed" data / the training target).
fn euler_trajectory(runtime: &SolveRuntime, x0: &[f64], params: &[f64]) -> Vec<Vec<f64>> {
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

#[test]
fn native_training_recovers_ground_truth_parameters() {
    let (runtime, solve_model) = build_runtime();
    assert_eq!(runtime.state_count, 2, "expected 2 states");
    assert_eq!(runtime.solver_count, 2, "pure ODE: solver_y == states");

    let x0 = vec![0.5_f64, -0.2];
    let settle = settle();

    // --- Resolve the a,b parameter slots (no hard-coding). ---
    let report = runtime.eval_parameter_jacobian(0.0, &x0, &solve_model.parameters, settle);
    assert!(report.error.is_none(), "param jacobian: {:?}", report.error);
    let slot_of = |name: &str| -> usize {
        let col = report
            .param_labels
            .iter()
            .position(|p| p == name)
            .unwrap_or_else(|| panic!("parameter {name} not found in {:?}", report.param_labels));
        report.param_slots[col]
    };
    let slot_a = slot_of("a");
    let slot_b = slot_of("b");

    // --- Ground-truth "observed" trajectory (a* = 0.7, b* = 0.3). ---
    let mut truth = solve_model.parameters.clone();
    truth[slot_a] = 0.7;
    truth[slot_b] = 0.3;
    let observed = euler_trajectory(&runtime, &x0, &truth);
    let targets: Vec<&[f64]> = observed.iter().map(|s| s.as_slice()).collect();

    // --- Perturbed starting guess (a = 0.3, b = 0.8). ---
    let mut params = solve_model.parameters.clone();
    params[slot_a] = 0.3;
    params[slot_b] = 0.8;

    let cfg = TrainConfig {
        grid: EulerGrid {
            t0: T0,
            h: H,
            steps: STEPS,
        },
        settle,
        epochs: 2000,
        lr: 1.0e-2,
    };

    let report = runtime
        .train_trajectory_mse(&x0, &mut params, &[slot_a, slot_b], &targets, &cfg)
        .expect("training should run");

    // --- Report: sparse loss curve + recovered params. ---
    let n = report.losses.len();
    eprintln!("epochs = {n}");
    for &i in &[0usize, n / 8, n / 4, n / 2, 3 * n / 4, n - 1] {
        eprintln!("  epoch {i:>5}: loss = {:.6e}", report.losses[i]);
    }
    let rec_a = params[slot_a];
    let rec_b = params[slot_b];
    eprintln!("initial_loss = {:.6e}", report.initial_loss);
    eprintln!("final_loss   = {:.6e}", report.final_loss);
    eprintln!("recovered a = {rec_a:.6} (truth 0.7), b = {rec_b:.6} (truth 0.3)");

    // --- Assertions: loss collapses, parameters recovered. ---
    assert!(
        report.final_loss < report.initial_loss / 100.0,
        "loss should drop by >100x: initial = {:.6e}, final = {:.6e}",
        report.initial_loss,
        report.final_loss
    );
    assert!(
        report.final_loss < 1.0e-6,
        "final loss should be < 1e-6, got {:.6e}",
        report.final_loss
    );
    assert!(
        (rec_a - 0.7).abs() < 1.0e-2,
        "recovered a = {rec_a}, expected ≈ 0.7"
    );
    assert!(
        (rec_b - 0.3).abs() < 1.0e-2,
        "recovered b = {rec_b}, expected ≈ 0.3"
    );

    // The report's written-back params must match the in-place params.
    assert_eq!(report.params[slot_a], rec_a);
    assert_eq!(report.params[slot_b], rec_b);
}
