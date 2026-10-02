use crate::{
    ast::{NamedDeBruijn, Term},
    global_uniq::next_uniq_id,
};

use super::value::{Env, Value};
use std::rc::Rc;

pub fn value_as_term(mut value: Value) -> Term<NamedDeBruijn> {
    // `Value` carries a manual, iterative `Drop` (see `machine/value.rs`), which makes the
    // compiler reject moving fields out of it by pattern. Matching on `&mut` instead costs
    // nothing: every field taken here is either `Copy`, a cheap `Rc` handle, or a vector we can
    // take outright.
    match &mut value {
        Value::Con(constant) => Term::Constant {
            value: Rc::clone(constant),
            uniq_id: next_uniq_id(),
        },
        Value::Builtin { runtime, fun, term_id } => {
            let mut term = Term::Builtin {
                fun: *fun,
                uniq_id: *term_id,
            };

            let forces = runtime.forces;
            let args = std::mem::take(&mut runtime.args);

            for _ in 0..forces {
                term = term.force();
            }

            for arg in args {
                term = term.apply(value_as_term(arg));
            }

            term
        }
        Value::Delay { body, env, term_id } => with_env(0, Rc::clone(env), Term::Delay {
            body: Rc::clone(body),
            uniq_id: *term_id,
        }),
        Value::Lambda { parameter_name, body, env, term_id } => with_env(
            0,
            Rc::clone(env),
            Term::Lambda {
                parameter_name: NamedDeBruijn {
                    text: parameter_name.text.clone(),
                    index: 0.into(),
                }
                .into(),
                body: Rc::clone(body),
                uniq_id: *term_id,
            },
        ),
        Value::Constr { tag, fields, term_id } => Term::Constr {
            tag: *tag,
            fields: std::mem::take(fields)
                .into_iter()
                .map(value_as_term)
                .collect(),
            uniq_id: *term_id,
        },
    }
}

fn with_env(lam_cnt: usize, env: Env, term: Term<NamedDeBruijn>) -> Term<NamedDeBruijn> {
    // `Term` carries a manual, iterative `Drop` (see `ast.rs`), which makes the compiler reject
    // moving fields out of it by pattern. Matching on a reference and cloning the `Rc` handles
    // costs the same, and the untouched arms still hand the original term straight back.
    match &term {
        Term::Var { name, uniq_id } => {
            let index: usize = name.index.into();

            if lam_cnt >= index {
                return term;
            }

            let name = Rc::clone(name);
            let uniq_id = *uniq_id;

            env.get::<usize>(env.len() - (index - lam_cnt))
                .cloned()
                .map_or(Term::Var { name, uniq_id }, value_as_term)
        }
        Term::Lambda { parameter_name, body, uniq_id } => {
            let parameter_name = Rc::clone(parameter_name);
            let uniq_id = *uniq_id;

            let body = with_env(lam_cnt + 1, env, body.as_ref().clone());

            Term::Lambda {
                parameter_name,
                body: body.into(),
                uniq_id,
            }
        }
        Term::Apply { function, argument, uniq_id } => {
            let uniq_id = *uniq_id;

            let function = with_env(lam_cnt, env.clone(), function.as_ref().clone());
            let argument = with_env(lam_cnt, env, argument.as_ref().clone());

            Term::Apply {
                function: function.into(),
                argument: argument.into(),
                uniq_id,
            }
        }

        Term::Delay { body, uniq_id } => {
            let uniq_id = *uniq_id;

            let delay = with_env(lam_cnt, env, body.as_ref().clone());

            Term::Delay {
                body: delay.into(),
                uniq_id,
            }
        }
        Term::Force { body, uniq_id } => {
            let uniq_id = *uniq_id;

            let force = with_env(lam_cnt, env, body.as_ref().clone());

            Term::Force {
                body: force.into(),
                uniq_id,
            }
        }
        _ => term,
    }
}
