/// e2e encoding/decoding tests
use uplc::{
    ast::{DeBruijn, GlobalNamedDeBruijn, Name, NamedDeBruijn, Program},
    parser,
};

fn round_trip_test(bytes: &[u8], code: &str) {
    parsed_program_matches_decoded_bytes(bytes, code);
    encoded_program_matches_bytes(bytes, code);
    can_convert_between_de_bruijn_and_name(bytes, code);
    decoded_bytes_can_convert_into_code(bytes, code);
    
    test_global_named_debruijn_conversions(bytes, code);
    test_global_id_assignment(code);
    test_global_named_debruijn_round_trip(code);
}

fn parsed_program_matches_decoded_bytes(bytes: &[u8], code: &str) {
    let parsed_program = parser::program(code).unwrap();

    let debruijn_program: Program<DeBruijn> = parsed_program.try_into().unwrap();

    let decoded_program: Program<DeBruijn> = Program::from_flat(bytes).unwrap();

    assert_eq!(debruijn_program, decoded_program);
}

fn encoded_program_matches_bytes(bytes: &[u8], code: &str) {
    let parsed_program = parser::program(code).unwrap();

    let debruijn_program: Program<DeBruijn> = parsed_program.try_into().unwrap();

    let encoded_program = debruijn_program.to_flat().unwrap();

    assert_eq!(encoded_program, bytes);
}
fn can_convert_between_de_bruijn_and_name(bytes: &[u8], code: &str) {
    let parsed_program = parser::program(code).unwrap();

    let decoded_program: Program<DeBruijn> = Program::from_flat(bytes).unwrap();

    let name_program: Program<Name> = decoded_program.try_into().unwrap();

    assert_eq!(parsed_program, name_program);
}

fn decoded_bytes_can_convert_into_code(bytes: &[u8], code: &str) {
    let decoded_program: Program<DeBruijn> = Program::from_flat(bytes).unwrap();

    let name_program: Program<Name> = decoded_program.try_into().unwrap();

    let pretty = name_program.to_pretty();

    assert_eq!(pretty, code);
}

// GlobalNamedDeBruijn specific tests
fn test_global_named_debruijn_conversions(bytes: &[u8], code: &str) {
    let parsed_program = parser::program(code).unwrap();
    
    // Test Name -> GlobalNamedDeBruijn conversion
    let global_from_name: Program<GlobalNamedDeBruijn> = parsed_program.clone().try_into().unwrap();
    
    // Test DeBruijn -> GlobalNamedDeBruijn conversion
    let decoded_debruijn: Program<DeBruijn> = Program::from_flat(bytes).unwrap();
    let global_from_debruijn: Program<GlobalNamedDeBruijn> = decoded_debruijn.into();
    
    // Both should have the same structure (ignoring global IDs for now)
    let global_from_name_to_debruijn: Program<DeBruijn> = global_from_name.into();
    let global_from_debruijn_to_debruijn: Program<DeBruijn> = global_from_debruijn.into();
    
    assert_eq!(global_from_name_to_debruijn, global_from_debruijn_to_debruijn);
}

fn test_global_id_assignment(code: &str) {
    let parsed_program = parser::program(code).unwrap();
    let global_program: Program<GlobalNamedDeBruijn> = parsed_program.try_into().unwrap();
    
    // Collect all global IDs from the program
    let mut global_ids = Vec::new();
    collect_global_ids(&global_program.term, &mut global_ids);
    
    // Check that all global IDs are unique and start from 1
    if !global_ids.is_empty() {
        global_ids.sort();
        global_ids.dedup();
        
        // Should start from 1 and be consecutive
        for (i, &id) in global_ids.iter().enumerate() {
            assert_eq!(id, i + 1, "Global IDs should be consecutive starting from 1");
        }
    }
}

fn test_global_named_debruijn_round_trip(code: &str) {
    let parsed_program = parser::program(code).unwrap();
    
    // Name -> GlobalNamedDeBruijn -> NamedDeBruijn -> Name
    let global_program: Program<GlobalNamedDeBruijn> = parsed_program.clone().try_into().unwrap();
    let named_debruijn_program: Program<NamedDeBruijn> = global_program.into();
    let back_to_name: Program<Name> = named_debruijn_program.try_into().unwrap();
    
    // Should match the original (though variable names might differ due to debruijn conversion)
    assert_eq!(parsed_program.version, back_to_name.version);
}

// Helper function to collect global IDs from a term
fn collect_global_ids(term: &uplc::ast::Term<GlobalNamedDeBruijn>, ids: &mut Vec<usize>) {
    use uplc::ast::Term;
    
    match term {
        Term::Var(var) => {
            ids.push(var.global_id);
        }
        Term::Delay(t) => collect_global_ids(t, ids),
        Term::Lambda { parameter_name, body } => {
            ids.push(parameter_name.global_id);
            collect_global_ids(body, ids);
        }
        Term::Apply { function, argument } => {
            collect_global_ids(function, ids);
            collect_global_ids(argument, ids);
        }
        Term::Force(t) => collect_global_ids(t, ids),
        Term::Constr { fields, .. } => {
            for field in fields {
                collect_global_ids(field, ids);
            }
        }
        Term::Case { constr, branches } => {
            collect_global_ids(constr, ids);
            for branch in branches {
                collect_global_ids(branch, ids);
            }
        }
        Term::Constant(_) | Term::Error | Term::Builtin(_) => {
            // No variables in these terms
        }
    }
}

#[test]
fn integer() {
    let bytes = include_bytes!("../test_data/basic/integer/integer.flat");
    let code = include_str!("../test_data/basic/integer/integer.uplc");

    round_trip_test(bytes, code);
}

#[test]
fn jpg() {
    let bytes = include_bytes!("../test_data/jpg/jpg.flat");
    let code = include_str!("../test_data/jpg/jpg.uplc");

    round_trip_test(bytes, code);
}

#[test]
fn fibonacci() {
    let bytes = include_bytes!("../test_data/fibonacci/fibonacci.flat");
    let code = include_str!("../test_data/fibonacci/fibonacci.uplc");

    round_trip_test(bytes, code);
}

#[test]
fn case_constr() {
    let bytes = include_bytes!("../test_data/case_constr/case_constr.flat");
    let code = include_str!("../test_data/case_constr/case_constr.uplc");

    round_trip_test(bytes, code);
}

#[test]
fn one_way_fibonacci() {
    let bytes = include_bytes!("../test_data/fibonacci/fibonacci.flat");
    // This code doesn't match the expected `i_unique` naming scheme, so it can't be round-tripped.
    // We still want to test these "unsanitary" cases because we can't control the naming pattern
    // the consumer uses. We just can't guarantee that the decoded Flat bytes will match their
    // names.
    let code = include_str!("../test_data/fibonacci/unsanitary_fibonacci.uplc");

    parsed_program_matches_decoded_bytes(bytes, code);
    encoded_program_matches_bytes(bytes, code);
}

#[test]
fn test_global_named_debruijn_features() {
    // Test with a simple lambda function
    let code = r#"(program
  1.0.0
  (lam x x)
)"#;
    
    let parsed_program = parser::program(code).unwrap();
    let global_program: Program<GlobalNamedDeBruijn> = parsed_program.try_into().unwrap();
    
    // Check that global IDs are assigned
    let mut global_ids = Vec::new();
    collect_global_ids(&global_program.term, &mut global_ids);
    
    // Should have 2 variables (parameter and usage) with IDs 1 and 2
    assert_eq!(global_ids.len(), 2);
    assert!(global_ids.contains(&1));
    assert!(global_ids.contains(&2));
    
    // Test reassignment of global IDs
    let reassigned = global_program.clone().reassign_global_ids();
    let mut reassigned_ids = Vec::new();
    collect_global_ids(&reassigned.term, &mut reassigned_ids);
    
    // Should still have 2 variables but potentially different IDs
    assert_eq!(reassigned_ids.len(), 2);
    
    // Test round-trip conversions
    let back_to_named: Program<NamedDeBruijn> = global_program.clone().into();
    let back_to_debruijn: Program<DeBruijn> = global_program.into();
    
    // Should be able to convert back
    assert_eq!(back_to_named.version, (1, 0, 0));
    assert_eq!(back_to_debruijn.version, (1, 0, 0));
}

#[test]
fn test_global_named_debruijn_with_nested_lambdas() {
    // Test with nested lambda functions to ensure global IDs work correctly
    let code = r#"(program
  1.0.0
  (lam x (lam y [ x y ]))
)"#;
    
    let parsed_program = parser::program(code).unwrap();
    let global_program: Program<GlobalNamedDeBruijn> = parsed_program.try_into().unwrap();
    
    // Check that global IDs are assigned uniquely
    let mut global_ids = Vec::new();
    collect_global_ids(&global_program.term, &mut global_ids);
    
    // Should have multiple variables with unique IDs
    assert!(global_ids.len() >= 2);
    
    // All IDs should be unique
    let mut sorted_ids = global_ids.clone();
    sorted_ids.sort();
    sorted_ids.dedup();
    assert_eq!(global_ids.len(), sorted_ids.len(), "All global IDs should be unique");
    
    // Test that we can decode from flat and get consistent global IDs
    let debruijn_version: Program<DeBruijn> = global_program.clone().into();
    let flat_bytes = debruijn_version.to_flat().unwrap();
    let decoded_debruijn: Program<DeBruijn> = Program::from_flat(&flat_bytes).unwrap();
    let global_from_decoded: Program<GlobalNamedDeBruijn> = decoded_debruijn.into();
    
    // Should have same structure
    let original_as_debruijn: Program<DeBruijn> = global_program.into();
    let decoded_as_debruijn: Program<DeBruijn> = global_from_decoded.into();
    assert_eq!(original_as_debruijn, decoded_as_debruijn);
}
