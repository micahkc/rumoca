//! Native parameter training for the rumoca solver.
//!
//! Fits a chosen subset of model parameters to observed trajectory data by
//! gradient descent (Adam) over the *discretize-then-optimize* loss
//! `L = (1/M) Σₙ Σᵢ (xₙᵢ − targetₙᵢ)²` (mean-squared error over every sampled
//! scalar state across all stored Euler steps). Each epoch re-integrates the
//! explicit-Euler trajectory forward, forms the per-step state cotangents
//! `ḡₙ = (2/M)·(xₙ − targetₙ)`, and calls the native trajectory adjoint
//! ([`SolveRuntime::trajectory_euler_adjoint`]) to obtain `dL/dθ` over the full
//! `p_scalars` slot space. Only the `trainable_slots` entries are gathered into
//! a dense trainable gradient and updated by Adam; the optimizer's updates are
//! written back into those same slots of `params` before the next epoch.
//!
//! This is the mechanism behind "train neural networks natively, no Python":
//! whether the trainable slots hold NN weights or physical constants, both are
//! just `parameter Real` slots in the same `p_scalars` space.

use crate::trajectory_adjoint::{EulerGrid, TrajectoryAdjointScratch};
use crate::{AlgebraicSettle, SolveRuntime};
use rumoca_solver::RuntimeSolveError;

/// Adam optimizer over a dense trainable slice.
///
/// Operates only on the `n_params` trainable values supplied to [`Adam::step`];
/// the caller is responsible for gathering the trainable gradient and scattering
/// the updated values back into the full parameter vector.
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate (step size).
    pub lr: f64,
    /// First-moment (mean) decay.
    pub beta1: f64,
    /// Second-moment (variance) decay.
    pub beta2: f64,
    /// Numerical-stability epsilon in the denominator.
    pub eps: f64,
    /// Step counter (used for bias correction).
    pub t: u64,
    /// First-moment estimate (length `n_params`).
    pub m: Vec<f64>,
    /// Second-moment estimate (length `n_params`).
    pub v: Vec<f64>,
}

impl Adam {
    /// Standard Adam defaults (`beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`).
    pub fn new(n_params: usize, lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            t: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
        }
    }

    /// One Adam update with bias correction. `params` and `grad` must both have
    /// length `n_params`; `params[i] -= lr · m̂ᵢ / (√v̂ᵢ + eps)`.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        debug_assert_eq!(params.len(), self.m.len());
        debug_assert_eq!(grad.len(), self.m.len());
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grad[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Configuration for [`SolveRuntime::train_trajectory_mse`].
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// Fixed-step Euler grid used for both the forward integration and adjoint.
    pub grid: EulerGrid,
    /// Algebraic-settle tolerances passed through to the adjoint kernel.
    pub settle: AlgebraicSettle,
    /// Number of optimizer epochs (one forward + adjoint + Adam step each).
    pub epochs: usize,
    /// Adam learning rate.
    pub lr: f64,
}

/// Outcome of a training run.
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// MSE loss after the final epoch's parameter update.
    pub final_loss: f64,
    /// MSE loss of the initial (pre-training) parameters.
    pub initial_loss: f64,
    /// Per-epoch loss (length `epochs`), recorded *before* each epoch's update.
    pub losses: Vec<f64>,
    /// The trained full parameter vector (a copy of the written-back `params`).
    pub params: Vec<f64>,
}

/// Reusable buffers for the per-epoch forward/adjoint/optimizer cycle, so the
/// steady-state training loop is allocation-free.
struct TrainScratch {
    /// Adjoint forward/backward scratch (states, lambda, vjp, der).
    adjoint: TrajectoryAdjointScratch,
    /// Predicted states, flattened `pred[n*state_count .. (n+1)*state_count]`.
    pred: Vec<f64>,
    /// Forward-pass derivative buffer (length `state_count`).
    der: Vec<f64>,
    /// Per-step cotangent storage, flattened like `pred`.
    cot: Vec<f64>,
    /// Dense trainable gradient (length `trainable_slots.len()`).
    grad_trainable: Vec<f64>,
    /// Dense trainable parameter values (length `trainable_slots.len()`).
    theta_trainable: Vec<f64>,
}

impl SolveRuntime {
    /// Fit `trainable_slots` (indices into the `p_scalars` slot space) so the
    /// explicit-Euler-integrated trajectory matches `targets`.
    ///
    /// `targets[n]` is the observed state vector at stored step `n`; the slice
    /// must have `grid.steps + 1` entries, each of length `state_count`. The
    /// loss is the mean-squared error over every sampled scalar state across all
    /// stored steps (`M = (steps + 1)·state_count`). `params` is the full
    /// parameter vector and is updated in place; its trained value is also
    /// returned in the report. Pure-ODE only (inherited from the adjoint kernel).
    pub fn train_trajectory_mse(
        &self,
        x0: &[f64],
        params: &mut [f64],
        trainable_slots: &[usize],
        targets: &[&[f64]],
        cfg: &TrainConfig,
    ) -> Result<TrainReport, RuntimeSolveError> {
        let state_count = self.state_count;
        let grid = cfg.grid;
        let stored = grid.steps + 1;
        self.validate_train_inputs(x0, params, trainable_slots, targets, stored)?;

        let m_samples = (stored * state_count) as f64;
        let inv_m = if m_samples > 0.0 {
            1.0 / m_samples
        } else {
            0.0
        };

        let mut scratch = TrainScratch {
            adjoint: TrajectoryAdjointScratch::default(),
            pred: vec![0.0; stored * state_count],
            der: vec![0.0; state_count],
            cot: vec![0.0; stored * state_count],
            grad_trainable: vec![0.0; trainable_slots.len()],
            theta_trainable: vec![0.0; trainable_slots.len()],
        };
        let mut adam = Adam::new(trainable_slots.len(), cfg.lr);
        let mut losses = Vec::with_capacity(cfg.epochs);

        let initial_loss = self.forward_and_loss(x0, params, targets, cfg, inv_m, &mut scratch)?;

        for _ in 0..cfg.epochs {
            // (a) forward pass + (b) loss and per-step cotangents ḡₙ = (2/M)(xₙ−tₙ).
            let loss = self.forward_and_loss(x0, params, targets, cfg, inv_m, &mut scratch)?;
            losses.push(loss);

            // (c) trajectory adjoint → dL/dθ over the full p_scalars slot space.
            let cotangents = step_views(&scratch.cot, stored, state_count);
            let gradient = self.trajectory_euler_adjoint(
                grid,
                x0,
                params,
                cfg.settle,
                &cotangents,
                &mut scratch.adjoint,
            )?;

            // (d) gather the trainable slots into a dense gradient + param vector.
            for (k, &slot) in trainable_slots.iter().enumerate() {
                scratch.grad_trainable[k] = gradient.grad_theta[slot];
                scratch.theta_trainable[k] = params[slot];
            }
            // (e) Adam step, then scatter the updates back into `params`.
            adam.step(&mut scratch.theta_trainable, &scratch.grad_trainable);
            for (k, &slot) in trainable_slots.iter().enumerate() {
                params[slot] = scratch.theta_trainable[k];
            }
        }

        let final_loss = self.forward_and_loss(x0, params, targets, cfg, inv_m, &mut scratch)?;

        Ok(TrainReport {
            final_loss,
            initial_loss,
            losses,
            params: params.to_vec(),
        })
    }

    /// Euler-integrate forward (mirroring the adjoint's forward pass), fill
    /// `scratch.pred`/`scratch.cot`, and return the MSE loss. The cotangents are
    /// `ḡₙ = (2/M)·(xₙ − targetₙ)`, the exact gradient of the MSE w.r.t. each
    /// stored state.
    fn forward_and_loss(
        &self,
        x0: &[f64],
        params: &[f64],
        targets: &[&[f64]],
        cfg: &TrainConfig,
        inv_m: f64,
        scratch: &mut TrainScratch,
    ) -> Result<f64, RuntimeSolveError> {
        let state_count = self.state_count;
        let grid = cfg.grid;
        scratch.pred[..state_count].copy_from_slice(x0);
        for n in 0..grid.steps {
            let t_n = grid.t0 + grid.h * n as f64;
            let (head, tail) = scratch.pred.split_at_mut((n + 1) * state_count);
            let x_n = &head[n * state_count..];
            self.eval_state_derivatives_into(
                t_n,
                x_n,
                params,
                cfg.settle.tol,
                cfg.settle.max_iters,
                &mut scratch.der,
            )?;
            let x_next = &mut tail[..state_count];
            for i in 0..state_count {
                x_next[i] = x_n[i] + grid.h * scratch.der[i];
            }
        }

        let mut sse = 0.0;
        for (n, &target_n) in targets.iter().enumerate() {
            let base = n * state_count;
            let pred_n = &scratch.pred[base..base + state_count];
            for i in 0..state_count {
                let diff = pred_n[i] - target_n[i];
                sse += diff * diff;
                scratch.cot[base + i] = 2.0 * inv_m * diff;
            }
        }
        Ok(sse * inv_m)
    }

    fn validate_train_inputs(
        &self,
        x0: &[f64],
        params: &[f64],
        trainable_slots: &[usize],
        targets: &[&[f64]],
        stored: usize,
    ) -> Result<(), RuntimeSolveError> {
        let state_count = self.state_count;
        if x0.len() != state_count {
            return Err(RuntimeSolveError::solve_ir(format!(
                "train: x0 has {} entries, expected state_count = {state_count}",
                x0.len()
            )));
        }
        if targets.len() != stored {
            return Err(RuntimeSolveError::solve_ir(format!(
                "train: {} targets, expected steps + 1 = {stored}",
                targets.len()
            )));
        }
        for (n, target) in targets.iter().enumerate() {
            if target.len() != state_count {
                return Err(RuntimeSolveError::solve_ir(format!(
                    "train: targets[{n}] has {} entries, expected state_count = {state_count}",
                    target.len()
                )));
            }
        }
        let p_scalars = self.model.problem.layout.p_scalars();
        for &slot in trainable_slots {
            if slot >= p_scalars {
                return Err(RuntimeSolveError::solve_ir(format!(
                    "train: trainable slot {slot} out of range for p_scalars = {p_scalars}"
                )));
            }
            if slot >= params.len() {
                return Err(RuntimeSolveError::solve_ir(format!(
                    "train: trainable slot {slot} out of range for params length {}",
                    params.len()
                )));
            }
        }
        Ok(())
    }
}

/// Borrow each stored step of a flattened `stored * state_count` buffer as a
/// `&[f64]` slice, for passing to the adjoint's `state_cotangents` parameter.
fn step_views(flat: &[f64], stored: usize, state_count: usize) -> Vec<&[f64]> {
    (0..stored)
        .map(|n| &flat[n * state_count..(n + 1) * state_count])
        .collect()
}
