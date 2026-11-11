use std::collections::HashMap;

use cc_problems::cc_system;
use cc_problems::cc_system::hamiltonian_spec::{
    DynOperatorSpec, DynPotentialSpec, HamiltonianSpec,
};
use cc_problems::cc_system::orbital_basis::{OrbitalBasisElements, OrbitalParity};
use cc_problems::cc_system::parameters::Parameters;
use cc_problems::hilbert_space::operator_mel;
use cc_problems::hilbert_space::space::{BasisElementsRef, SpaceBasis, SpaceElement};
use cc_problems::problems::rotor_atom_in_b_field::{RotorParams, RotorSpinParams};
use cc_problems::spin_algebra::ops::{
    red_first_subsystem_mel_factor, red_reduced_harmonics_mel, wigner_eckart_factor,
};
use cc_problems::spin_algebra::{SpinLike, SpinMagLike, get_spin_pair_magnitudes, spin};
use cc_problems::unit_systems::atomic_units;
use cc_problems::unit_systems::quantities::Scalar;
use cc_problems::unit_systems::quantities::phys_quantities::{
    ElectricDipole, ElectricField, MagneticField, Mass,
};
use cc_system::hamiltonian_spec::OperatorSpec;
use cc_system::parameters::{ParamIds, ParameterRegistry, TypedParamId};
use cc_system::{Operator, param_ids};
use serde::Deserialize;

use cc_calculations::calculations::DynCalc;
use cc_calculations::calculations::adiabats::{AdiabatsCalc, adiabats_calc_mods};
use cc_calculations::calculations::bound_states::{
    BoundStateCalc, bound_states_calc_mods, bound_states_calc_search_mods,
};
use cc_calculations::calculations::dependence::DependenceCalc;
use cc_calculations::calculations::hamiltonian_terms::hamiltonian_terms_calc;
use cc_calculations::calculations::levels::EnergyLevelsCalc;
use cc_calculations::calculations::modifications::{ModifyRegistry, NumberCalcMod, ScalarParamMod};
use cc_calculations::calculations::resonances::ResonancesCalc;
use cc_calculations::calculations::scattering::{ScatteringCalc, scattering_calc_mods};
use cc_calculations::modify_recipe;
use cc_calculations::potential_scalings::PesPolarizationScalingMod;
use cc_problems::atomic_systems::atom_operators::AtomParams;
use cc_problems::atomic_systems::diatom_operators::SpinRotationSpec;
use cc_problems::atomic_systems::interactions::pes::{
    PesPolarizationScalings, PesPolarizationSpec, PesPolarizations,
};
use cc_problems::atomic_systems::interactions::{Interactions, SpinConfiguration};
use cc_problems::atomic_systems::rotor_atom_basis::{TRAMRotorAtomBasis, TRAMRotorAtomBasisRecipe};
use cc_problems::cc_calculations::problems::Problem;

use crate::calculations::{ReactiveScatteringCalc, reactive_scattering_calc_mods};

#[derive(Debug, Clone, Deserialize, cc_system::Parameters, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct RotorAtomInBEFieldParams {
    #[serde(default)]
    pub b_field: Scalar<MagneticField>,
    #[serde(default)]
    pub e_field: Scalar<ElectricField>,
    pub red_mass: Scalar<Mass>,

    #[parameter(nested)]
    #[serde(default)]
    pub rotor_spins: RotorSpinParams,

    #[parameter(nested)]
    #[serde(default)]
    pub atom: AtomParams,

    #[parameter(nested)]
    pub rotor: RotorParams,
    pub rotor_dipole_moment: Scalar<ElectricDipole>,

    pub pes: PesPolarizations,
    #[serde(default)]
    pub scalings: PesPolarizationScalings,

    #[serde(default)]
    pub spin_orbit: Option<Interactions>,
}

pub fn rotor_atom_in_be_field_basis_mods<P, C>() -> ModifyRegistry<P, C>
where
    P: Problem<BasisRecipe = TRAMRotorAtomBasisRecipe> + 'static,
    C: Send + Sync + 'static,
{
    ModifyRegistry::from([
        (
            "n_max",
            modify_recipe!(
                |n_max| NumberCalcMod::new(n_max, |r: &mut P::BasisRecipe, n| r.tram.n_max = n)
            ),
        ),
        (
            "l_max",
            modify_recipe!(
                |l_max| NumberCalcMod::new(l_max, |r: &mut P::BasisRecipe, l| r.tram.l_max = l)
            ),
        ),
        (
            "n_tot_max",
            modify_recipe!(
                |n_max| NumberCalcMod::new(n_max, |r: &mut P::BasisRecipe, n| r.tram.n_tot_max = n)
            ),
        ),
    ])
}

pub fn rotor_atom_in_be_field_params_mods<P, C>() -> ModifyRegistry<P, C>
where
    P: Problem<Params = RotorAtomInBEFieldParams> + 'static,
    C: Send + Sync + 'static,
{
    let ids = RotorAtomInBEFieldParams::ids();
    ModifyRegistry::from([
        (
            "magnetic_field",
            modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field)),
        ),
        (
            "electric_field",
            modify_recipe!(|b| ScalarParamMod::new(b, ids.e_field)),
        ),
        (
            "red_mass",
            modify_recipe!(|x| ScalarParamMod::new(x, ids.red_mass)),
        ),
        (
            "pes_scaling",
            modify_recipe!(|x| PesPolarizationScalingMod::new(x, ids.pes, ids.scalings)),
        ),
    ])
}

pub fn rotor_atom_in_be_field_params_search_mods<P, C>() -> ModifyRegistry<P, C>
where
    P: Problem<Params = RotorAtomInBEFieldParams> + 'static,
    C: Send + Sync + 'static,
{
    let ids = RotorAtomInBEFieldParams::ids();
    ModifyRegistry::from([
        (
            "magnetic_field",
            modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field)),
        ),
        (
            "red_mass",
            modify_recipe!(|x| ScalarParamMod::new(x, ids.red_mass)),
        ),
        (
            "pes_scaling",
            modify_recipe!(|x| PesPolarizationScalingMod::new(x, ids.pes, ids.scalings)),
        ),
    ])
}

pub fn rotor_atom_levels_be_field_scan() -> Box<dyn DynCalc<RotorAtomInBEFieldProblem>> {
    let levels_calc = EnergyLevelsCalc::default();

    let ids = RotorAtomInBEFieldParams::ids();
    let registry = ModifyRegistry::from([
        (
            "magnetic_field",
            modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field)),
        ),
        (
            "electric_field",
            modify_recipe!(|b| ScalarParamMod::new(b, ids.e_field)),
        ),
        ]);
    Box::new(DependenceCalc::new(levels_calc, registry))
}

pub fn rotor_atom_adiabats_be_field_scan() -> Box<dyn DynCalc<RotorAtomInBEFieldProblem>> {
    let ids = RotorAtomInBEFieldParams::ids();
    let adiabats_calc = AdiabatsCalc::new(ids.red_mass);

    let registry = adiabats_calc_mods();
    Box::new(DependenceCalc::new(adiabats_calc, registry))
}

pub fn rotor_atom_scattering_be_field_scan() -> Box<dyn DynCalc<RotorAtomInBEFieldProblem>> {
    let ids = RotorAtomInBEFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);

    let registry = rotor_atom_in_be_field_basis_mods()
        .extend(rotor_atom_in_be_field_params_mods())
        .extend(scattering_calc_mods());
    Box::new(DependenceCalc::new(scattering_calc, registry))
}

pub fn rotor_atom_bound_states_be_field_scan() -> Box<dyn DynCalc<RotorAtomInBEFieldProblem>> {
    let ids = RotorAtomInBEFieldParams::ids();
    let registry =
        rotor_atom_in_be_field_params_search_mods().extend(bound_states_calc_search_mods());
    let bound_state_calc = BoundStateCalc::new(ids.red_mass, registry);

    let registry = rotor_atom_in_be_field_basis_mods()
        .extend(rotor_atom_in_be_field_params_mods())
        .extend(bound_states_calc_mods());
    Box::new(DependenceCalc::new(bound_state_calc, registry))
}

pub fn rotor_atom_resonances_scan() -> Box<dyn DynCalc<RotorAtomInBEFieldProblem>> {
    let ids = RotorAtomInBEFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);
    let registry =
        rotor_atom_in_be_field_params_search_mods().extend(bound_states_calc_search_mods());
    let bound_state_calc = BoundStateCalc::new(ids.red_mass, registry);

    let resonances_calc = ResonancesCalc::new(scattering_calc, bound_state_calc);

    let registry = rotor_atom_in_be_field_basis_mods().extend(rotor_atom_in_be_field_params_mods());
    Box::new(DependenceCalc::new(resonances_calc, registry))
}

pub fn rotor_atom_be_field_reactive_scattering() -> Box<dyn DynCalc<RotorAtomInBEFieldProblem>> {
    let ids = RotorAtomInBEFieldParams::ids();
    let scattering_calc = ReactiveScatteringCalc::new(ids.red_mass);

    let registry = rotor_atom_in_be_field_basis_mods()
        .extend(rotor_atom_in_be_field_params_mods())
        .extend(reactive_scattering_calc_mods());

    Box::new(DependenceCalc::new(scattering_calc, registry))
}

#[derive(Default)]
pub struct RotorAtomInBEFieldProblem {
    pub calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>>,
}

impl RotorAtomInBEFieldProblem {
    pub fn new() -> Self {
        let mut calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>> = HashMap::default();
        calculations.extend([
            ("hamiltonian terms".into(), hamiltonian_terms_calc()),
            ("levels scan".into(), rotor_atom_levels_be_field_scan()),
            ("adiabats scan".into(), rotor_atom_adiabats_be_field_scan()),
            (
                "scattering scan".into(),
                rotor_atom_scattering_be_field_scan(),
            ),
            (
                "bound states scan".into(),
                rotor_atom_bound_states_be_field_scan(),
            ),
            ("resonances scan".into(), rotor_atom_resonances_scan()),
            (
                "reactive scattering scan".into(),
                rotor_atom_be_field_reactive_scattering(),
            ),
        ]);

        Self { calculations }
    }
}

impl Problem for RotorAtomInBEFieldProblem {
    type BasisRecipe = TRAMRotorAtomBasisRecipe;
    type Params = RotorAtomInBEFieldParams;

    const NAME: &str = "rotor + atom in BE field";

    const DESCRIPTION: &str = "Rotor + Atom in external magnetic field";

    fn build(basis_recipe: &Self::BasisRecipe, params: &Self::Params) -> HamiltonianSpec {
        let param_ids = Self::Params::ids();

        let mut basis = SpaceBasis::default();
        let rotor_atom = TRAMRotorAtomBasis::new(basis_recipe, &mut basis);

        let projection_filter = |m, x: SpaceElement| {
            rotor_atom.filter(|((s_r, i_ra, i_rb), (s_a, i_a), n_tot)| {
                s_r.m() + i_ra.m() + i_rb.m() + s_a.m() + i_a.m() + n_tot.m() == m
            })(x)
        };
        let projection_filter = |x: SpaceElement| {
            if let Some(proj) = basis_recipe.projection {
                projection_filter(proj, x)
            } else {
                true
            }
        };
        let l_filter = |x: SpaceElement| match basis_recipe.l_parity {
            OrbitalParity::All => true,
            OrbitalParity::Even => {
                rotor_atom.filter(|(_, _, n_tot)| n_tot.pair.1.is_multiple_of(2))(x)
            }
            OrbitalParity::Odd => {
                rotor_atom.filter(|(_, _, n_tot)| !n_tot.pair.1.is_multiple_of(2))(x)
            }
        };

        let elements = basis.get_filtered_basis(|x| projection_filter(x) && l_filter(x));
        let elements =
            OrbitalBasisElements::new(elements, rotor_atom.tram.tram, |n_tot| n_tot.pair.1);

        let mut hamiltonian_spec = HamiltonianSpec::new(elements);

        hamiltonian_spec.add_operators(vec![
            (
                "atom.hifi",
                DynOperatorSpec::new(rotor_atom.atom.hifi(param_ids.atom.a_hifi)),
            ),
            (
                "atom.zeeman_e",
                DynOperatorSpec::new(
                    rotor_atom
                        .atom
                        .zeeman_e(param_ids.b_field, param_ids.atom.g_e),
                ),
            ),
            (
                "atom.zeeman_n",
                DynOperatorSpec::new(
                    rotor_atom
                        .atom
                        .zeeman_n(param_ids.b_field, param_ids.atom.g_n),
                ),
            ),
            (
                "rotor.energy",
                DynOperatorSpec::new(rotor_atom.rot_energy(param_ids.rotor.rot_const)),
            ),
            (
                "rotor.energy_distortion",
                DynOperatorSpec::new(
                    rotor_atom.rot_energy_distortion(param_ids.rotor.rot_distortion),
                ),
            ),
            (
                "rotor.hifi_a",
                DynOperatorSpec::new(rotor_atom.rotor_a.hifi(param_ids.rotor_spins.a_hifi_a)),
            ),
            (
                "rotor.hifi_b",
                DynOperatorSpec::new(rotor_atom.rotor_b.hifi(param_ids.rotor_spins.a_hifi_b)),
            ),
            (
                "rotor.zeeman_e",
                DynOperatorSpec::new(
                    rotor_atom
                        .rotor_a
                        .zeeman_e(param_ids.b_field, param_ids.rotor_spins.g_e),
                ),
            ),
            (
                "rotor.zeeman_n_a",
                DynOperatorSpec::new(
                    rotor_atom
                        .rotor_a
                        .zeeman_n(param_ids.b_field, param_ids.rotor_spins.g_n_a),
                ),
            ),
            (
                "rotor.zeeman_n_b",
                DynOperatorSpec::new(
                    rotor_atom
                        .rotor_b
                        .zeeman_n(param_ids.b_field, param_ids.rotor_spins.g_n_b),
                ),
            ),
            (
                "rotor.aniso_hifi_a",
                DynOperatorSpec::new(rotor_atom.aniso_hifi_a(param_ids.rotor_spins.c_hifi_a)),
            ),
            (
                "rotor.aniso_hifi_b",
                DynOperatorSpec::new(rotor_atom.aniso_hifi_b(param_ids.rotor_spins.c_hifi_b)),
            ),
            (
                "rotor.e_rot",
                DynOperatorSpec::new(rotor_atom.spin_e_rot(param_ids.rotor.spin_e_rot)),
            ),
            (
                "rotor.n_a_rot",
                DynOperatorSpec::new(rotor_atom.spin_n_a_rot(param_ids.rotor.spin_n_a_rot)),
            ),
            (
                "rotor.n_b_rot",
                DynOperatorSpec::new(rotor_atom.spin_n_b_rot(param_ids.rotor.spin_n_b_rot)),
            ),
            (
                "rotor.stark",
                DynOperatorSpec::new(StarkSpec::new(
                    param_ids.e_field,
                    param_ids.rotor_dipole_moment,
                    move |b| {
                        operator_mel!(b, [rotor_atom.tram.tram], |[nl]| {
                            if nl.bra.spin.m == nl.ket.spin.m && nl.bra.pair.1 == nl.ket.pair.1 {
                                -wigner_eckart_factor(nl, spin!(1, 0))
                                    * red_first_subsystem_mel_factor(
                                        nl.map(|x| x.as_spin_pair_mag()),
                                        1,
                                    )
                                    * red_reduced_harmonics_mel(nl.map(|x| x.pair.0), 1)
                            } else {
                                0.
                            }
                        })
                    },
                )),
            ),
        ]);

        let s_r = basis_recipe.rotor.s;
        let s_a = basis_recipe.atom.s;
        let polarizations = get_spin_pair_magnitudes([s_r], [s_a]);
        hamiltonian_spec.add_potentials(polarizations.into_iter().flat_map(|p| {
            let pes = &params.pes.0[&SpinConfiguration::Spin(p.s())];
            let s_tot = p.s();

            pes.components().into_iter().map(move |lambda| {
                (
                    format!("pes_s_{s_tot}_lambda_{lambda}"),
                    DynPotentialSpec::new(PesPolarizationSpec {
                        s_tot,
                        lambda,
                        pes: param_ids.pes,
                        scalings: param_ids.scalings,
                        masking: rotor_atom.pes_polarization_legendre_component(lambda, s_tot),
                    }),
                )
            })
        }));
        hamiltonian_spec.add_potentials([(
            "spin_rot_coupling",
            DynPotentialSpec::new(SpinRotationSpec {
                so_curve: param_ids.spin_orbit,
                masking: rotor_atom.second_order_so(),
            }),
        )]);

        hamiltonian_spec
    }

    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>> {
        &self.calculations
    }
}

pub struct StarkSpec<F: Fn(BasisElementsRef) -> Operator> {
    pub e_field: TypedParamId<Scalar<ElectricField>>,
    pub dipole: TypedParamId<Scalar<ElectricDipole>>,
    operator: F,
}

impl<F: Fn(BasisElementsRef) -> Operator> StarkSpec<F> {
    pub fn new(
        e_field: TypedParamId<Scalar<ElectricField>>,
        dipole: TypedParamId<Scalar<ElectricDipole>>,
        operator: F,
    ) -> Self {
        Self {
            e_field,
            dipole,
            operator,
        }
    }
}

impl<F: Fn(BasisElementsRef) -> Operator + Send + Sync> OperatorSpec for StarkSpec<F> {
    fn build_params(&self) -> ParamIds {
        param_ids![]
    }

    fn matrix(&self, elements: BasisElementsRef, _params: &ParameterRegistry) -> Operator {
        (self.operator)(elements)
    }

    fn coupling_params(&self) -> ParamIds {
        param_ids![self.e_field.vanish(), self.dipole.vanish()]
    }

    fn coupling(&self, params: &ParameterRegistry) -> f64 {
        let converter = atomic_units::units_converter();

        let e_field = converter.scalar_value(params.get(self.e_field));
        let dipole = converter.scalar_value(params.get(self.dipole));

        e_field * dipole
    }
}
