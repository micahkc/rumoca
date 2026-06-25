//! Native discrete (trajectory) adjoint: backpropagating through an explicit
//! Euler integration of a pure-ODE model must compute the exact gradient of the
//! Euler-discretized loss `L = Σₙ (xₙ² + yₙ²)` w.r.t. the model parameters.
//!
//! Two oracles gate the adjoint gradient:
//!  1. **Forward sensitivities** integrated with the *identical* Euler stepper
//!     (`eval_forward_sensitivity_column_into`): `dL/dp = Σₙ ḡₙᵀ sₙ`. Since both
//!     adjoint and forward sensitivity differentiate the same discretized loss,
//!     they must agree to round-off (`< 1e-7`).
//!  2. **Central finite difference** of the Euler-integrated loss w.r.t. each
//!     parameter (`ε = 1e-6`), to FD tolerance `1e-4 + 1e-4·|fd|`.

use rumoca::Compiler;
use rumoca_eval_solve::{
    AlgebraicLinearization, AlgebraicSettle, EulerGrid, SolveRuntime, TrajectoryAdjointScratch,
};
use rumoca_sim::SimOptions;

// Pure ODE, strictly smooth and bounded so finite differences agree:
//   der(x) = sin(a*x) + b*y^2
//   der(y) = tanh(a*y) - b*x
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

/// Euler-integrate the trajectory and return `L = Σₙ (xₙ² + yₙ²)` over the
/// stored states `x₀ … x_N` (the same loss the adjoint differentiates).
fn euler_loss(runtime: &SolveRuntime, x0: &[f64], params: &[f64]) -> f64 {
    let settle = settle();
    let mut state = x0.to_vec();
    let mut der = vec![0.0; runtime.state_count];
    let mut loss = state.iter().map(|v| v * v).sum::<f64>();
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
        loss += state.iter().map(|v| v * v).sum::<f64>();
    }
    loss
}

/// Oracle 1: integrate the state and one sensitivity column `sₙ = ∂xₙ/∂p` with
/// the SAME Euler stepper, accumulating `dL/dp = Σₙ ḡₙᵀ sₙ` where `ḡₙ = 2·xₙ`.
fn forward_sensitivity_grad(
    runtime: &SolveRuntime,
    x0: &[f64],
    params: &[f64],
    param_slot: usize,
) -> f64 {
    let settle = settle();
    let n_state = runtime.state_count;
    let mut state = x0.to_vec();
    let mut sens = vec![0.0; n_state]; // ∂x₀/∂p = 0
    let mut der = vec![0.0; n_state];
    let mut sens_rhs = vec![0.0; n_state];

    // n = 0 contribution: ḡ₀ᵀ s₀ = 0 (s₀ = 0), but keep the general form.
    let mut grad: f64 = state.iter().zip(&sens).map(|(x, s)| 2.0 * x * s).sum();

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
        // ds/dt = ∂f/∂x·s + ∂f/∂p, evaluated at the current (state, sens).
        runtime
            .eval_forward_sensitivity_column_into(
                AlgebraicLinearization {
                    t: t_n,
                    params,
                    settle,
                },
                &state,
                &sens,
                param_slot,
                &mut sens_rhs,
            )
            .expect("forward sensitivity RHS");
        for i in 0..n_state {
            state[i] += H * der[i];
            sens[i] += H * sens_rhs[i];
        }
        // Accumulate the loss-cotangent pairing for the new stored state.
        grad += state
            .iter()
            .zip(&sens)
            .map(|(x, s)| 2.0 * x * s)
            .sum::<f64>();
    }
    grad
}

#[test]
fn trajectory_euler_adjoint_matches_forward_sensitivity_and_fd() {
    let (runtime, solve_model) = build_runtime();
    assert_eq!(runtime.state_count, 2, "expected 2 states");
    assert_eq!(runtime.solver_count, 2, "pure ODE: solver_y == states");

    let x0 = vec![0.5_f64, -0.2];
    let params = solve_model.parameters.clone();
    let settle = settle();

    // --- Run the forward pass to collect stored states, then build ḡₙ = 2·xₙ. ---
    // We re-derive the stored trajectory here so the cotangents line up with the
    // adjoint's internal store-all forward pass (same Euler stepper).
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(STEPS + 1);
    {
        let mut state = x0.clone();
        states.push(state.clone());
        let mut der = vec![0.0; runtime.state_count];
        for n in 0..STEPS {
            let t_n = T0 + H * n as f64;
            runtime
                .eval_state_derivatives_into(
                    t_n,
                    &state,
                    &params,
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
    }
    let cotangent_storage: Vec<Vec<f64>> = states
        .iter()
        .map(|x| x.iter().map(|v| 2.0 * v).collect())
        .collect();
    let state_cotangents: Vec<&[f64]> = cotangent_storage.iter().map(|c| c.as_slice()).collect();

    // --- Native trajectory adjoint. ---
    let mut scratch = TrajectoryAdjointScratch::default();
    let grid = EulerGrid {
        t0: T0,
        h: H,
        steps: STEPS,
    };
    let gradient = runtime
        .trajectory_euler_adjoint(grid, &x0, &params, settle, &state_cotangents, &mut scratch)
        .expect("trajectory adjoint should evaluate");
    assert_eq!(gradient.loss_steps, STEPS + 1);
    assert_eq!(gradient.grad_x0.len(), runtime.state_count);

    // --- Resolve the a,b parameter slots (no hard-coding). ---
    let report = runtime.eval_parameter_jacobian(0.0, &x0, &params, settle);
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

    // --- Oracle 1: forward sensitivities, identical Euler stepper. ---
    let fwd_a = forward_sensitivity_grad(&runtime, &x0, &params, slot_a);
    let fwd_b = forward_sensitivity_grad(&runtime, &x0, &params, slot_b);

    let adj_a = gradient.grad_theta[slot_a];
    let adj_b = gradient.grad_theta[slot_b];

    eprintln!("grad_theta[a]: adjoint={adj_a}, forward_sens={fwd_a}");
    eprintln!("grad_theta[b]: adjoint={adj_b}, forward_sens={fwd_b}");
    eprintln!("grad_x0 = {:?}", gradient.grad_x0);

    assert!(
        (adj_a - fwd_a).abs() < 1.0e-7,
        "dL/da adjoint vs forward-sensitivity mismatch: adjoint={adj_a}, forward={fwd_a}, \
         delta={}",
        (adj_a - fwd_a).abs()
    );
    assert!(
        (adj_b - fwd_b).abs() < 1.0e-7,
        "dL/db adjoint vs forward-sensitivity mismatch: adjoint={adj_b}, forward={fwd_b}, \
         delta={}",
        (adj_b - fwd_b).abs()
    );

    // --- Oracle 2: central finite difference of the Euler-integrated loss. ---
    let eps = 1.0e-6;
    let fd_param = |slot: usize| -> f64 {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[slot] += eps;
        minus[slot] -= eps;
        let lp = euler_loss(&runtime, &x0, &plus);
        let lm = euler_loss(&runtime, &x0, &minus);
        (lp - lm) / (2.0 * eps)
    };
    let fd_a = fd_param(slot_a);
    let fd_b = fd_param(slot_b);
    eprintln!("grad_theta[a]: adjoint={adj_a}, fd={fd_a}");
    eprintln!("grad_theta[b]: adjoint={adj_b}, fd={fd_b}");
    assert!(
        (adj_a - fd_a).abs() <= 1.0e-4 + 1.0e-4 * fd_a.abs(),
        "dL/da adjoint vs FD: adjoint={adj_a}, fd={fd_a}"
    );
    assert!(
        (adj_b - fd_b).abs() <= 1.0e-4 + 1.0e-4 * fd_b.abs(),
        "dL/db adjoint vs FD: adjoint={adj_b}, fd={fd_b}"
    );

    // --- Sanity: the gradient is non-vacuous. ---
    assert!(
        adj_a.abs() > 1.0e-6 && adj_b.abs() > 1.0e-6,
        "gradient should be non-vacuous: dL/da={adj_a}, dL/db={adj_b}"
    );
}
