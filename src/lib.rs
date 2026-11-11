use cc_problems::problems::{
    AvailableProblems, diatom_in_b_field::DiatomInBFieldProblem,
    rotor_atom_in_b_field::RotorAtomInBFieldProblem,
};

use crate::{
    calculations::{diatom_reactive_scattering, rotor_atom_b_field_reactive_scattering},
    system::RotorAtomInBEFieldProblem,
};

pub mod calculations;
pub mod system;

pub fn available_problems() -> AvailableProblems {
    let mut problems = AvailableProblems::default();

    let mut diatom = DiatomInBFieldProblem::new();
    diatom.calculations.insert(
        "reactive scattering scan".into(),
        diatom_reactive_scattering(),
    );
    problems.insert(diatom);

    let mut rotor_atom = RotorAtomInBFieldProblem::new();
    rotor_atom.calculations.insert(
        "reactive scattering scan".into(),
        rotor_atom_b_field_reactive_scattering(),
    );
    problems.insert(rotor_atom);

    let rotor_atom = RotorAtomInBEFieldProblem::new();
    problems.insert(rotor_atom);

    problems
}
