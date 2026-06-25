//! Native discrete (trajectory) adjoint for the rumoca solver.
//!
//! Backpropagates through a fixed-step **explicit Euler** time integration to
//! compute `dL/dθ` (and `dL/dx₀`) for a loss `L = Σₙ gₙ(xₙ)` over the stored
//! states of a pure-ODE model. This is the *discretize-then-optimize* gradient:
//! it is the exact gradient of the Euler-discretized loss, so it matches a
//! forward-sensitivity integration that uses the identical Euler stepper to
//! round-off (the validation oracle), and a central finite difference of the
//! Euler-integrated loss to FD tolerance.
//!
//! ## Math (explicit Euler discrete adjoint)
//! Forward step `x_{n+1} = xₙ + h·f(tₙ, xₙ, θ)`. With per-sample state cotangent
//! `ḡₙ = ∂gₙ/∂xₙ` (caller-supplied), the backward recurrence for `n = N-1 … 0`,
//! taking `μ = λ_{n+1}` and one reverse-VJP kernel call at `(tₙ, xₙ)` yielding
//! `vjp = [vjp_x | vjp_p] = (∂f/∂[x|θ])ᵀ μ`, is
//! ```text
//! λ_N      = ḡ_N
//! λ_n      = ḡ_n + λ_{n+1} + h·vjp_x        // Aₙᵀλ = (I + h·∂f/∂x)ᵀ λ
//! grad_θ  += h·vjp_p                          // Bₙᵀλ = h·(∂f/∂θ)ᵀ λ
//! ```
//! At the end `grad_x0 = λ_0`.
//!
//! ## Scope (v1)
//! Pure ODEs only (`solver_count == state_count`): the reverse kernel
//! ([`SolveRuntime::reverse_state_derivative_vjp`]) does not chain through the
//! algebraic projection and refuses models with solver algebraics; that error is
//! surfaced unchanged here. The forward pass stores *all* states (store-all);
//! checkpointing is a v2 concern.

use crate::{AlgebraicLinearization, AlgebraicSettle, SolveRuntime};
use rumoca_solver::RuntimeSolveError;

/// Fixed-step explicit-Euler integration grid. Stores `steps + 1` states
/// (`x₀ … x_N`), so the caller supplies `steps + 1` state cotangents.
#[derive(Debug, Clone, Copy)]
pub struct EulerGrid {
    /// Initial time `t₀`.
    pub t0: f64,
    /// Fixed step size `h`.
    pub h: f64,
    /// Number of Euler steps `N` (the grid stores `N + 1` states).
    pub steps: usize,
}

/// Reusable, allocation-free scratch for [`SolveRuntime::trajectory_euler_adjoint`].
/// Buffers are cleared (retaining capacity) and resized at the start of each
/// call, so a single instance can be threaded through a training loop without
/// re-allocating.
#[derive(Default)]
pub struct TrajectoryAdjointScratch {
    /// All stored states, flattened: `states[n*state_count .. (n+1)*state_count]`
    /// is `xₙ` for `n = 0 … steps`.
    states: Vec<f64>,
    /// Current adjoint `λ` (length `state_count`).
    lambda: Vec<f64>,
    /// Per-step reverse VJP output `[vjp_x (solver_count) | vjp_p (p_scalars)]`.
    vjp: Vec<f64>,
    /// Forward state-derivative buffer `f(tₙ, xₙ, θ)` (length `state_count`).
    der: Vec<f64>,
}

/// Result of a trajectory adjoint: the loss gradient over the full parameter
/// slot space and over the initial state.
#[derive(Debug, Clone)]
pub struct TrajectoryGradient {
    /// `dL/dθ` over the full `p_scalars` slot space. Map parameter names to slots
    /// via the parameter-Jacobian report's `param_labels`/`param_slots`.
    pub grad_theta: Vec<f64>,
    /// `dL/dx₀` (length `state_count`).
    pub grad_x0: Vec<f64>,
    /// Number of stored states that contributed loss cotangents (`steps + 1`).
    pub loss_steps: usize,
}

impl SolveRuntime {
    /// Native explicit-Euler trajectory (discrete) adjoint: backpropagate through
    /// the Euler-integrated trajectory to compute `dL/dθ` and `dL/dx₀` for the
    /// loss `L = Σₙ gₙ(xₙ)`.
    ///
    /// `x0` is the initial state (length `state_count`); `params` is the full
    /// parameter vector; `settle` tolerances are passed through to each per-step
    /// reverse-VJP linearization. `state_cotangents[n]` is `ḡₙ = ∂gₙ/∂xₙ` for the
    /// stored state `xₙ`, so the slice must have exactly `steps + 1` entries, each
    /// of length `state_count` (use a zero vector where a step contributes no loss).
    ///
    /// Pure-ODE only: a model with solver algebraics is rejected by the underlying
    /// [`Self::reverse_state_derivative_vjp`] kernel and that error is surfaced here.
    pub fn trajectory_euler_adjoint(
        &self,
        grid: EulerGrid,
        x0: &[f64],
        params: &[f64],
        settle: AlgebraicSettle,
        state_cotangents: &[&[f64]],
        scratch: &mut TrajectoryAdjointScratch,
    ) -> Result<TrajectoryGradient, RuntimeSolveError> {
        let state_count = self.state_count;
        if x0.len() != state_count {
            return Err(RuntimeSolveError::solve_ir(format!(
                "trajectory adjoint: x0 has {} entries, expected state_count = {state_count}",
                x0.len()
            )));
        }
        let stored = grid.steps + 1;
        if state_cotangents.len() != stored {
            return Err(RuntimeSolveError::solve_ir(format!(
                "trajectory adjoint: {} state cotangents, expected steps + 1 = {stored}",
                state_cotangents.len()
            )));
        }
        for (n, cot) in state_cotangents.iter().enumerate() {
            if cot.len() != state_count {
                return Err(RuntimeSolveError::solve_ir(format!(
                    "trajectory adjoint: state_cotangents[{n}] has {} entries, expected \
                     state_count = {state_count}",
                    cot.len()
                )));
            }
        }
        let p_scalars = self.model.problem.layout.p_scalars();
        let vjp_len = self.solver_count + p_scalars;

        // --- Forward pass: Euler-step, storing every state x_0 … x_N. ---
        scratch.states.clear();
        scratch.states.resize(stored * state_count, 0.0);
        scratch.der.clear();
        scratch.der.resize(state_count, 0.0);
        scratch.states[..state_count].copy_from_slice(x0);
        for n in 0..grid.steps {
            let t_n = grid.t0 + grid.h * n as f64;
            let (head, tail) = scratch.states.split_at_mut((n + 1) * state_count);
            let x_n = &head[n * state_count..];
            self.eval_state_derivatives_into(
                t_n,
                x_n,
                params,
                settle.tol,
                settle.max_iters,
                &mut scratch.der,
            )?;
            let x_next = &mut tail[..state_count];
            for i in 0..state_count {
                x_next[i] = x_n[i] + grid.h * scratch.der[i];
            }
        }

        // --- Backward pass: discrete adjoint recurrence. ---
        let mut grad_theta = vec![0.0_f64; p_scalars];
        scratch.lambda.clear();
        scratch.lambda.resize(state_count, 0.0);
        scratch.vjp.clear();
        scratch.vjp.resize(vjp_len, 0.0);

        // λ_N = ḡ_N.
        scratch.lambda.copy_from_slice(state_cotangents[grid.steps]);

        // n = N-1 … 0: one kernel call at (t_n, x_n) with μ = λ_{n+1}.
        for n in (0..grid.steps).rev() {
            let t_n = grid.t0 + grid.h * n as f64;
            let x_n = &scratch.states[n * state_count..(n + 1) * state_count];
            let lin = AlgebraicLinearization {
                t: t_n,
                params,
                settle,
            };
            // vjp = (∂f/∂[x|θ])ᵀ · λ_{n+1} (kernel fills `out` from zero).
            self.reverse_state_derivative_vjp(lin, x_n, &scratch.lambda, &mut scratch.vjp)?;
            let (vjp_x, vjp_p) = scratch.vjp.split_at(self.solver_count);

            // grad_θ += h · vjp_p.
            for (g, &v) in grad_theta.iter_mut().zip(vjp_p) {
                *g += grid.h * v;
            }
            // λ_n = ḡ_n + λ_{n+1} + h · vjp_x.
            let g_n = state_cotangents[n];
            for i in 0..state_count {
                scratch.lambda[i] = g_n[i] + scratch.lambda[i] + grid.h * vjp_x[i];
            }
        }

        // grad_x0 = λ_0.
        let grad_x0 = scratch.lambda.clone();
        Ok(TrajectoryGradient {
            grad_theta,
            grad_x0,
            loss_steps: stored,
        })
    }
}
