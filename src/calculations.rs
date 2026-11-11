use std::f64::consts::PI;

use cc_calculations::{
    calculations::{
        DynCalc, Modified, SingleCalc,
        dependence::DependenceCalc,
        modifications::{FloatCalcMod, ModifyRegistry, ScalarCalcMod},
        scattering::{ScatteringCalc, TwoSegmentPropagator},
    },
    modify_recipe,
    problems::{self, Problem},
};
use cc_math_utils::bessel::{
    riccati_i_log_deriv, riccati_j_deriv, riccati_k_log_deriv, riccati_n_deriv,
};
use cc_problems::{
    coupled_chan::{
        cc_propagator::{Boundary, ConnectionWatcher, Direction},
        coupling::{Asymptote, CollisionParams, CollisionWMatrix},
        multi_channel::{Matrix, WMatrix},
    },
    hilbert_space::faer::{
        Mat, Side, c64, complex::Complex64, diag::Diag, linalg::solvers::DenseSolveCore,
    },
    problems::{
        diatom_in_b_field::{
            DiatomInBFieldParams, DiatomInBFieldProblem, diatom_in_b_field_basis_mods,
            diatom_in_b_field_params_mods,
        },
        rotor_atom_in_b_field::{
            RotorAtomInBFieldParams, RotorAtomInBFieldProblem, rotor_atom_in_b_field_basis_mods,
            rotor_atom_in_b_field_params_mods,
        },
    },
    unit_systems::{
        atomic_units,
        quantities::{
            Scalar,
            phys_quantities::{Energy, Length, Mass},
        },
    },
};
use cc_system::parameters::{Parameters, TypedParamId};
use serde::{Deserialize, Serialize};

// Based on "Cold Collision Dynamics of NH Radicals" Janssen

#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ReactiveScatteringCalcInput {
    pub entrance: usize,
    pub energy: Scalar<Energy>,

    pub r_min: Scalar<Length>,
    pub r_max: Scalar<Length>,

    #[serde(flatten)]
    pub propagator: TwoSegmentPropagator,
}

pub fn reactive_scattering_calc_mods<P: Problem + 'static>()
-> ModifyRegistry<P, ReactiveScatteringCalcInput> {
    ModifyRegistry::from([
        (
            "energy",
            modify_recipe!(
                |e| ScalarCalcMod::new(e, |r: &mut ReactiveScatteringCalcInput, v| r.energy = v)
            ),
        ),
        (
            "r_min",
            modify_recipe!(
                |x| ScalarCalcMod::new(x, |r: &mut ReactiveScatteringCalcInput, v| r.r_min = v)
            ),
        ),
        (
            "r_max",
            modify_recipe!(
                |x| ScalarCalcMod::new(x, |r: &mut ReactiveScatteringCalcInput, v| r.r_max = v)
            ),
        ),
        (
            "step_scaling",
            modify_recipe!(
                |x| FloatCalcMod::new(x, |r: &mut ReactiveScatteringCalcInput, v| r
                    .propagator
                    .scale_step(v))
            ),
        ),
    ])
}

pub struct ReactiveScatteringCalc<P> {
    scattering: ScatteringCalc<P>,
}

impl<P: Problem> ReactiveScatteringCalc<P> {
    pub fn new(mass_id: TypedParamId<Scalar<Mass>>) -> Self {
        Self {
            scattering: ScatteringCalc::new(mass_id),
        }
    }
}

impl<P: Problem> SingleCalc for ReactiveScatteringCalc<P> {
    type P = P;
    type CalcInput = ReactiveScatteringCalcInput;
    type Data = ReactiveSMatrixData;

    fn calculate(
        &self,
        modified: &mut Modified<P, ReactiveScatteringCalcInput>,
        _problem: &P,
    ) -> impl IntoIterator<Item = anyhow::Result<ReactiveSMatrixData>> {
        let registry = modified.system.param_registry();
        let converter = atomic_units::units_converter();

        let blocks = modified.system.angular_blocks();
        let collision_params = CollisionParams {
            mass: converter.scalar_value(registry.get(self.scattering.mass)),
            energy: converter.scalar_value(&modified.calc_input.energy),
            entrance: modified.calc_input.entrance,
        };
        let asymptote = Asymptote::new_angular_blocks(blocks, collision_params);
        let w_matrix = CollisionWMatrix::new(modified.system.coupling(), asymptote);

        let r_min = converter.scalar_value(&modified.calc_input.r_min);
        let r_max = converter.scalar_value(&modified.calc_input.r_max);

        let boundary_out = Boundary {
            r_start: r_min,
            direction: Direction::Outwards,
            value: w_matrix.id().to_owned(),
            derivative: 0. * w_matrix.id(),
        };
        let boundary_in = Boundary {
            r_start: r_max,
            direction: Direction::Inwards,
            value: w_matrix.id().to_owned(),
            derivative: 0. * w_matrix.id(),
        };

        let (f_0, f_n) = regular_solutions(
            &modified.calc_input.propagator,
            boundary_out,
            r_max,
            &w_matrix,
        );
        let f_0 = Mat::from_fn(f_0.nrows(), f_0.ncols(), |i, j| c64::from(f_0[(i, j)]));
        let f_n = Mat::from_fn(f_n.nrows(), f_n.ncols(), |i, j| c64::from(f_n[(i, j)]));

        let (g_n, g_0) = regular_solutions(
            &modified.calc_input.propagator,
            boundary_in,
            r_max,
            &w_matrix,
        );
        let g_0 = Mat::from_fn(g_0.nrows(), g_0.ncols(), |i, j| c64::from(g_0[(i, j)]));
        let g_n = Mat::from_fn(g_n.nrows(), g_n.ncols(), |i, j| c64::from(g_n[(i, j)]));

        let size = w_matrix.size();
        let asymptote = &w_matrix.asymptote();
        let levels = asymptote.levels();

        let is_open_channel_n = levels
            .asymptote
            .iter()
            .map(|&val| val < asymptote.energy)
            .collect::<Vec<bool>>();
        let momenta_n: Vec<f64> = levels
            .asymptote
            .iter()
            .map(|&val| {
                (2.0 * asymptote.system_params().mass * (asymptote.energy - val).abs()).sqrt()
            })
            .collect();

        let transf: Mat<c64> = if let Some(transformation) = asymptote.transformation() {
            Mat::from_fn(transformation.nrows(), transformation.ncols(), |i, j| {
                c64::from(transformation[(i, j)])
            })
        } else {
            Mat::identity(size, size)
        };

        let mut outflux_n = Diag::zeros(size);
        let mut outflux_deriv_n = Diag::zeros(size);
        let mut influx_n = Diag::zeros(size);
        let mut influx_deriv_n = Diag::zeros(size);

        for i in 0..size {
            let momentum = momenta_n[i];
            let l = levels.l[i];
            if is_open_channel_n[i] {
                let (j_riccati, j_deriv_riccati) = riccati_j_deriv(l, momentum * r_max);
                let (n_riccati, n_deriv_riccati) = riccati_n_deriv(l, momentum * r_max);

                influx_n[i] = c64::new(j_riccati, n_riccati) / momentum.sqrt();
                influx_deriv_n[i] = c64::new(j_deriv_riccati, n_deriv_riccati) * momentum.sqrt();

                outflux_n[i] = c64::new(j_riccati, -n_riccati) / momentum.sqrt();
                outflux_deriv_n[i] = c64::new(j_deriv_riccati, -n_deriv_riccati) * momentum.sqrt();
            } else {
                let ratio_i = riccati_i_log_deriv(l, momentum * r_max);
                let ratio_k = riccati_k_log_deriv(l, momentum * r_max);

                influx_n[i] = c64::from(1.);
                influx_deriv_n[i] = c64::from(ratio_i) * momentum;

                outflux_n[i] = c64::from(1.);
                outflux_deriv_n[i] = c64::from(ratio_k) * momentum;
            }
        }

        let influx_n = &transf * influx_n;
        let influx_deriv_n = &transf * influx_deriv_n;
        let outflux_n = &transf * outflux_n;
        let outflux_deriv_n = &transf * outflux_deriv_n;

        let mut values = Matrix::zeros(size, size);
        w_matrix.value_inplace(r_min, &mut values);

        let levels = values.self_adjoint_eigen(Side::Lower).unwrap();

        let is_open_channel_0 = levels
            .S()
            .column_vector()
            .iter()
            .map(|&val| val > 0.)
            .collect::<Vec<bool>>();
        let momenta_0: Vec<f64> = levels
            .S()
            .column_vector()
            .iter()
            .map(|&val| val.abs().sqrt())
            .collect();

        let mut outflux_0 = Diag::zeros(size);
        let mut outflux_deriv_0 = Diag::zeros(size);

        for i in 0..size {
            let momentum = momenta_0[i];
            if is_open_channel_0[i] {
                outflux_0[i] = c64::from_polar(1. / momentum.sqrt(), -momentum * r_min);
                outflux_deriv_0[i] = -c64::i() * c64::from_polar(momentum.sqrt(), -momentum * r_min);
            } else {
                // outflux_0[i] = c64::from(momentum * r_min);
                // outflux_deriv_0[i] = c64::from(momentum * r_min) * momentum;

                outflux_0[i] = c64::from(1.);
                outflux_deriv_0[i] = c64::from(momentum);
            }
        }

        let outflux_0 = &transf * outflux_0;
        let outflux_deriv_0 = &transf * outflux_deriv_0;

        let g_o_terms = &g_n
            * &outflux_deriv_0
            * (&g_0 * &outflux_deriv_0 - &outflux_0)
                .partial_piv_lu()
                .inverse()
            * f_0;
        let denominator = (&f_n * &outflux_deriv_n - &outflux_n) - &g_o_terms * &outflux_deriv_n;
        let numerator = (&f_n * &influx_deriv_n - &influx_n) - &g_o_terms * &influx_deriv_n;

        let s_matrix = -denominator.partial_piv_lu().inverse() * numerator;

        let open_channel_count = is_open_channel_n.iter().filter(|val| **val).count();
        // let mut red_s_matrix = Mat::<c64>::zeros(open_channel_count, open_channel_count);
        let mut red_s_matrix = Mat::<c64>::zeros(size, open_channel_count);
        let mut open_channel_indices = Vec::with_capacity(open_channel_count);

        let mut j_full = 0;
        for j in 0..open_channel_count {
            while !is_open_channel_n[j_full] {
                j_full += 1
            }
            open_channel_indices.push(j_full);

            // let mut i_full = 0;
            // for i in 0..open_channel_count {
            // while !is_open_channel_n[i_full] {
            //     i_full += 1
            // }

            // red_s_matrix[(i, j)] = s_matrix[(i_full, j_full)];
            // i_full += 1;
            // }
            for i in 0..size {
                red_s_matrix[(i, j)] = s_matrix[(i, j_full)];
            }

            j_full += 1;
        }

        let entrance = asymptote.system_params().entrance;
        let entrance_red = is_open_channel_n
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .enumerate()
            .find(|(_, (i, _))| *i == entrance)
            .expect("Closed entrance channel")
            .0;

        let momenta_red: Vec<f64> = momenta_n
            .into_iter()
            .zip(is_open_channel_n)
            .filter(|(_, i)| *i)
            .map(|(p, _)| p)
            .collect();
        let momentum_entrance = momenta_red[entrance_red];

        let s_entrance = s_matrix[(entrance, entrance_red)];
        let s_length =
            1. / Complex64::new(0., momentum_entrance) * (1. - s_entrance) / (1. + s_entrance);
        let cross_sects_to: Vec<f64> = (0..size)
            .into_iter()
            .map(|i| {
                if i == entrance {
                    PI / momentum_entrance.powi(2) * (1.0 - s_entrance).norm_sqr()
                } else {
                    PI / momentum_entrance.powi(2) * (s_matrix[(i, entrance_red)]).norm_sqr()
                }
            })
            .collect();
        let inelastic_cross_sect = cross_sects_to
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != entrance)
            .map(|(_, x)| x)
            .sum();

        let s_matrix_data = ReactiveSMatrixData {
            s_matrix_vec: problems::mat_as_nested_vec(s_matrix.as_ref()),
            momenta: momenta_red,
            entrance_nr: entrance_red,
            cross_sects_to,
            inelastic_cross_sect,
            s_length,
            open_channel_indices,
        };

        [Ok(s_matrix_data)]
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReactiveSMatrixData {
    pub s_matrix_vec: Vec<Vec<Complex64>>,
    pub momenta: Vec<f64>,
    pub entrance_nr: usize,
    pub open_channel_indices: Vec<usize>,

    pub cross_sects_to: Vec<f64>,
    pub inelastic_cross_sect: f64,
    pub s_length: Complex64,
}

fn regular_solutions(
    propagator: &TwoSegmentPropagator,
    boundary: Boundary<Matrix>,
    r_end: f64,
    w_matrix: &impl WMatrix<f64>,
) -> (Matrix, Matrix) {
    let converter = atomic_units::units_converter();

    let mut regular_solution = RegularSolutionInit::new(w_matrix.size());

    let sol = match (
        &propagator.segment_change,
        &propagator.long_range_propagator,
    ) {
        (Some(r_change), Some(lr_propagator)) => {
            let r_change = converter.scalar_value(r_change);

            let mut propagator = propagator.propagator.get_propagator(boundary, w_matrix);
            propagator.insert_watcher(&mut regular_solution);
            let nodes = propagator.nodes();

            let solution = propagator.propagate_to(r_change).to_owned();
            drop(propagator);

            let mut lr_propagator = lr_propagator.transfer_propagator(solution, nodes, w_matrix);
            lr_propagator.insert_watcher(&mut regular_solution);

            lr_propagator.propagate_to(r_end).to_owned()
        }
        _ => {
            let mut propagator = propagator.propagator.get_propagator(boundary, w_matrix);
            propagator.insert_watcher(&mut regular_solution);

            propagator.propagate_to(r_end).to_owned()
        }
    };

    let f_end = sol.sol.0.partial_piv_lu().inverse();

    (regular_solution.f_start * &f_end, f_end)
}

/// solution f for which f'(r_start) = 0, f'(r_end) = 1
/// without
pub struct RegularSolutionInit {
    pub f_start: Matrix,
}

impl RegularSolutionInit {
    pub fn new(size: usize) -> Self {
        Self {
            f_start: Matrix::identity(size, size),
        }
    }
}

impl ConnectionWatcher<Matrix> for RegularSolutionInit {
    fn connection_at(&mut self, _r: f64, connection: &Matrix) {
        self.f_start = &self.f_start * connection
    }
}

pub fn diatom_reactive_scattering() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let scattering_calc = ReactiveScatteringCalc::new(ids.red_mass);

    let registry = diatom_in_b_field_basis_mods()
        .extend(diatom_in_b_field_params_mods())
        .extend(reactive_scattering_calc_mods());

    Box::new(DependenceCalc::new(scattering_calc, registry))
}

pub fn rotor_atom_b_field_reactive_scattering() -> Box<dyn DynCalc<RotorAtomInBFieldProblem>> {
    let ids = RotorAtomInBFieldParams::ids();
    let scattering_calc = ReactiveScatteringCalc::new(ids.red_mass);

    let registry = rotor_atom_in_b_field_basis_mods()
        .extend(rotor_atom_in_b_field_params_mods())
        .extend(reactive_scattering_calc_mods());

    Box::new(DependenceCalc::new(scattering_calc, registry))
}
