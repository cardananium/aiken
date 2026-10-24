use crate::{
    ast::{NamedDeBruijn, Term},
    global_uniq::next_uniq_id,
};

use super::value::{Env, Value};
use std::rc::Rc;

/// What is left to do while turning a value back into a term.
///
/// Discharging follows a value as deep as it goes -- `Constr` fields, builtin arguments, closure
/// bodies, and the environment values a body refers to, which are themselves discharged -- and
/// every one of those is as deep as the script makes it (see the `Drop` impl on `Term` in `ast.rs`
/// for the full rationale). So the work is kept on an explicit stack: `Task`s on `todo`, finished
/// terms on `done`. A step that needs its children first pushes a `Build*` task under them; since
/// `todo` is LIFO, children go on in reverse of the order they are processed in, and come off
/// `done` in reverse of the order they were finished.
///
/// The order is exactly the one of the recursive definition this replaces (`value_as_term` and
/// `with_env`), because it is observable: fresh term ids are handed out along the way, for each
/// discharged constant and for every `force` and application rebuilt around a builtin.
enum Task {
    /// Discharge a value.
    Value(Value),
    /// Substitute the values of `env` for the free variables of `term`, under `lam_cnt` lambdas.
    WithEnv {
        lam_cnt: usize,
        env: Env,
        term: Term<NamedDeBruijn>,
    },
    /// Apply the term on top of `done` (an argument) to the one below it (a builtin, partially
    /// applied so far).
    BuildBuiltinApply,
    /// A `constr` of the last `arity` terms finished.
    BuildConstr {
        tag: usize,
        arity: usize,
        term_id: isize,
    },
    /// A lambda around the term on top of `done`.
    BuildLambda {
        parameter_name: Rc<NamedDeBruijn>,
        uniq_id: isize,
    },
    /// An application of the two terms on top of `done`: function below, argument on top.
    BuildApply {
        uniq_id: isize,
    },
    BuildDelay {
        uniq_id: isize,
    },
    BuildForce {
        uniq_id: isize,
    },
}

pub fn value_as_term(value: Value) -> Term<NamedDeBruijn> {
    let mut todo = vec![Task::Value(value)];
    let mut done: Vec<Term<NamedDeBruijn>> = Vec::new();

    while let Some(task) = todo.pop() {
        match task {
            Task::Value(value) => discharge_value(value, &mut todo, &mut done),
            Task::WithEnv { lam_cnt, env, term } => {
                with_env(lam_cnt, env, term, &mut todo, &mut done)
            }
            Task::BuildBuiltinApply => {
                let argument = done.pop().expect("argument discharged");
                let builtin = done.pop().expect("builtin discharged");

                done.push(builtin.apply(argument));
            }
            Task::BuildConstr {
                tag,
                arity,
                term_id,
            } => {
                let fields = done.split_off(done.len() - arity);

                done.push(Term::Constr {
                    tag,
                    fields,
                    uniq_id: term_id,
                });
            }
            Task::BuildLambda {
                parameter_name,
                uniq_id,
            } => {
                let body = done.pop().expect("body discharged");

                done.push(Term::Lambda {
                    parameter_name,
                    body: body.into(),
                    uniq_id,
                });
            }
            Task::BuildApply { uniq_id } => {
                let argument = done.pop().expect("argument discharged");
                let function = done.pop().expect("function discharged");

                done.push(Term::Apply {
                    function: function.into(),
                    argument: argument.into(),
                    uniq_id,
                });
            }
            Task::BuildDelay { uniq_id } => {
                let body = done.pop().expect("body discharged");

                done.push(Term::Delay {
                    body: body.into(),
                    uniq_id,
                });
            }
            Task::BuildForce { uniq_id } => {
                let body = done.pop().expect("body discharged");

                done.push(Term::Force {
                    body: body.into(),
                    uniq_id,
                });
            }
        }
    }

    done.pop().expect("the discharged root")
}

fn discharge_value(mut value: Value, todo: &mut Vec<Task>, done: &mut Vec<Term<NamedDeBruijn>>) {
    // `Value` carries a manual, iterative `Drop` (see `machine/value.rs`), which makes the
    // compiler reject moving fields out of it by pattern. Matching on `&mut` instead costs
    // nothing: every field taken here is either `Copy`, a cheap `Rc` handle, or a vector we can
    // take outright.
    match &mut value {
        Value::Con(constant) => done.push(Term::Constant {
            value: Rc::clone(constant),
            uniq_id: next_uniq_id(),
        }),
        Value::Builtin {
            runtime,
            fun,
            term_id,
        } => {
            let mut term = Term::Builtin {
                fun: *fun,
                uniq_id: *term_id,
            };

            for _ in 0..runtime.forces {
                term = term.force();
            }

            done.push(term);

            // Each argument is discharged, then applied, before the next one is looked at.
            for arg in std::mem::take(&mut runtime.args).into_iter().rev() {
                todo.push(Task::BuildBuiltinApply);
                todo.push(Task::Value(arg));
            }
        }
        Value::Delay { body, env, term_id } => todo.push(Task::WithEnv {
            lam_cnt: 0,
            env: Rc::clone(env),
            term: Term::Delay {
                body: Rc::clone(body),
                uniq_id: *term_id,
            },
        }),
        Value::Lambda {
            parameter_name,
            body,
            env,
            term_id,
        } => todo.push(Task::WithEnv {
            lam_cnt: 0,
            env: Rc::clone(env),
            term: Term::Lambda {
                parameter_name: NamedDeBruijn {
                    text: parameter_name.text.clone(),
                    index: 0.into(),
                }
                .into(),
                body: Rc::clone(body),
                uniq_id: *term_id,
            },
        }),
        Value::Constr {
            tag,
            fields,
            term_id,
        } => {
            let fields = std::mem::take(fields);

            todo.push(Task::BuildConstr {
                tag: *tag,
                arity: fields.len(),
                term_id: *term_id,
            });
            todo.extend(fields.into_iter().rev().map(Task::Value));
        }
    }
}

fn with_env(
    lam_cnt: usize,
    env: Env,
    term: Term<NamedDeBruijn>,
    todo: &mut Vec<Task>,
    done: &mut Vec<Term<NamedDeBruijn>>,
) {
    // `Term` carries a manual, iterative `Drop` (see `ast.rs`), which makes the compiler reject
    // moving fields out of it by pattern. Matching on a reference and cloning the `Rc` handles
    // costs the same, and the untouched arms still hand the original term straight back.
    match &term {
        Term::Var { name, uniq_id } => {
            let index: usize = name.index.into();

            if lam_cnt >= index {
                done.push(term);
                return;
            }

            let name = Rc::clone(name);
            let uniq_id = *uniq_id;

            match env.get::<usize>(env.len() - (index - lam_cnt)).cloned() {
                Some(value) => todo.push(Task::Value(value)),
                None => done.push(Term::Var { name, uniq_id }),
            }
        }
        Term::Lambda {
            parameter_name,
            body,
            uniq_id,
        } => {
            todo.push(Task::BuildLambda {
                parameter_name: Rc::clone(parameter_name),
                uniq_id: *uniq_id,
            });
            todo.push(Task::WithEnv {
                lam_cnt: lam_cnt + 1,
                env,
                term: body.as_ref().clone(),
            });
        }
        Term::Apply {
            function,
            argument,
            uniq_id,
        } => {
            todo.push(Task::BuildApply { uniq_id: *uniq_id });
            todo.push(Task::WithEnv {
                lam_cnt,
                env: env.clone(),
                term: argument.as_ref().clone(),
            });
            todo.push(Task::WithEnv {
                lam_cnt,
                env,
                term: function.as_ref().clone(),
            });
        }
        Term::Delay { body, uniq_id } => {
            todo.push(Task::BuildDelay { uniq_id: *uniq_id });
            todo.push(Task::WithEnv {
                lam_cnt,
                env,
                term: body.as_ref().clone(),
            });
        }
        Term::Force { body, uniq_id } => {
            todo.push(Task::BuildForce { uniq_id: *uniq_id });
            todo.push(Task::WithEnv {
                lam_cnt,
                env,
                term: body.as_ref().clone(),
            });
        }
        _ => done.push(term),
    }
}
