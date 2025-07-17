use crate::{
    ast::{NamedDeBruijn, Term},
    global_uniq::next_uniq_id,
};

use super::value::{Env, Value};

pub fn value_as_term(value: Value) -> Term<NamedDeBruijn> {
    match value {
        Value::Con(constant) => Term::Constant {
            value: constant,
            uniq_id: next_uniq_id(),
        },
        Value::Builtin { runtime, fun, term_id } => {
            let mut term = Term::Builtin {
                fun: fun,
                uniq_id: term_id,
            };

            for _ in 0..runtime.forces {
                term = term.force();
            }

            for arg in runtime.args {
                term = term.apply(value_as_term(arg));
            }

            term
        }
        Value::Delay { body, env, term_id } => with_env(0, env, Term::Delay {
            body,
            uniq_id: term_id,
        }),
        Value::Lambda { parameter_name, body, env, term_id } => with_env(
            0,
            env,
            Term::Lambda {
                parameter_name: NamedDeBruijn {
                    text: parameter_name.text.clone(),
                    index: 0.into(),
                }
                .into(),
                body,
                uniq_id: term_id,
            },
        ),
        Value::Constr { tag, fields, term_id } => Term::Constr {
            tag,
            fields: fields.into_iter().map(value_as_term).collect(),
            uniq_id: term_id,
        },
    }
}

fn with_env(lam_cnt: usize, env: Env, term: Term<NamedDeBruijn>) -> Term<NamedDeBruijn> {
    match term {
        Term::Var { name, uniq_id } => {
            let index: usize = name.index.into();

            if lam_cnt >= index {
                Term::Var { name, uniq_id }
            } else {
                env.get::<usize>(env.len() - (index - lam_cnt))
                    .cloned()
                    .map_or(Term::Var { name, uniq_id }, value_as_term)
            }
        }
        Term::Lambda { parameter_name, body, uniq_id } => {
            let body = with_env(lam_cnt + 1, env, body.as_ref().clone());

            Term::Lambda {
                parameter_name,
                body: body.into(),
                uniq_id,
            }
        }
        Term::Apply { function, argument, uniq_id } => {
            let function = with_env(lam_cnt, env.clone(), function.as_ref().clone());
            let argument = with_env(lam_cnt, env, argument.as_ref().clone());

            Term::Apply {
                function: function.into(),
                argument: argument.into(),
                uniq_id,
            }
        }

        Term::Delay { body, uniq_id } => {
            let delay = with_env(lam_cnt, env, body.as_ref().clone());

            Term::Delay {
                body: delay.into(),
                uniq_id,
            }
        }
        Term::Force { body, uniq_id } => {
            let force = with_env(lam_cnt, env, body.as_ref().clone());

            Term::Force {
                body: force.into(),
                uniq_id,
            }
        }
        rest => rest,
    }
}
