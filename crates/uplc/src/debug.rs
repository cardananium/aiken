//! `Debug` output in the derived format, written without recursion.
//!
//! Terms, constants, types and machine values nest as deep as the script makes them (see the
//! `Drop` impl on `Term` in `ast.rs`), and the derived `Debug` follows that nesting one stack frame
//! per level, plain and `{:#?}` alike -- which matters because error messages embed it. The types
//! that nest describe each node as an [`Item`] instead, and [`write`] lays the items out off an
//! explicit stack, exactly as `Formatter::debug_struct`, `debug_tuple` and `debug_list` would.
//!
//! A field of another type is a [`Item::Leaf`], formatted by that type's own `Debug`. Those
//! formatters are iterative as well, and no type here contains the one that contains it, so the
//! leaves add a bounded number of frames, not one per level.

use std::fmt::{self, Write};

/// One piece of `Debug` output still to be written.
pub(crate) enum Item<'a, N> {
    /// A value of the type being formatted, described by the function given to [`write`].
    Node(&'a N),
    /// A value formatted by its own `Debug`.
    Leaf(&'a dyn fmt::Debug),
    /// `name(a, b)`, as `Formatter::debug_tuple` writes it; just `name` without fields.
    Tuple(&'static str, Vec<Item<'a, N>>),
    /// `name { a: x }`, as `Formatter::debug_struct` writes it.
    Struct(&'static str, Vec<(&'static str, Item<'a, N>)>),
    /// `[a, b]`, as `Formatter::debug_list` writes it.
    List(Vec<Item<'a, N>>),
    Text(&'static str),
    /// One level further in, for the lines `{:#?}` gives each field and item.
    Indent,
    Dedent,
}

/// Write `root` the way its derived `Debug` would, `describe` saying what each node holds.
pub(crate) fn write<'a, N>(
    root: &'a N,
    f: &mut fmt::Formatter<'_>,
    describe: impl Fn(&'a N) -> Item<'a, N>,
) -> fmt::Result {
    let pretty = f.alternate();

    let mut out = Indented {
        f,
        depth: 0,
        line_start: false,
    };

    let mut stack = vec![Item::Node(root)];

    while let Some(item) = stack.pop() {
        match item {
            Item::Node(node) => stack.push(describe(node)),
            Item::Leaf(leaf) => {
                if pretty {
                    write!(out, "{leaf:#?}")?;
                } else {
                    write!(out, "{leaf:?}")?;
                }
            }
            Item::Tuple(name, fields) => {
                if !fields.is_empty() {
                    stack.push(Item::Text(")"));
                    queue_members(&mut stack, pretty, fields.into_iter().map(|f| (None, f)));
                    stack.push(Item::Text(if pretty { "(\n" } else { "(" }));
                }
                stack.push(Item::Text(name));
            }
            Item::Struct(name, fields) => {
                if !fields.is_empty() {
                    stack.push(Item::Text(if pretty { "}" } else { " }" }));
                    let opening = if pretty { " {\n" } else { " { " };
                    queue_members(
                        &mut stack,
                        pretty,
                        fields.into_iter().map(|(label, f)| (Some(label), f)),
                    );
                    stack.push(Item::Text(opening));
                }
                stack.push(Item::Text(name));
            }
            Item::List(items) => {
                stack.push(Item::Text("]"));
                let any = !items.is_empty();
                queue_members(&mut stack, pretty, items.into_iter().map(|i| (None, i)));
                if pretty && any {
                    stack.push(Item::Text("\n"));
                }
                stack.push(Item::Text("["));
            }
            Item::Text(text) => out.write_str(text)?,
            Item::Indent => out.depth += 1,
            Item::Dedent => out.depth -= 1,
        }
    }

    Ok(())
}

/// Queue the fields of a tuple or struct, or the items of a list, last one first: separated by
/// `, ` in the plain format, each on a line of its own and followed by `,` with `{:#?}`.
fn queue_members<'a, N>(
    stack: &mut Vec<Item<'a, N>>,
    pretty: bool,
    members: impl DoubleEndedIterator<Item = (Option<&'static str>, Item<'a, N>)>
    + ExactSizeIterator,
) {
    for (i, (label, member)) in members.enumerate().rev() {
        if pretty {
            stack.push(Item::Dedent);
            stack.push(Item::Text(",\n"));
        }

        stack.push(member);

        if let Some(label) = label {
            stack.push(Item::Text(": "));
            stack.push(Item::Text(label));
        }

        if pretty {
            stack.push(Item::Indent);
        } else if i > 0 {
            stack.push(Item::Text(", "));
        }
    }
}

/// Writes through to a formatter the way the standard library's `{:#?}` builders do: every line
/// started at nesting depth `n` is indented by `4 * n` spaces.
struct Indented<'a, 'b> {
    f: &'a mut fmt::Formatter<'b>,
    depth: usize,
    line_start: bool,
}

impl Write for Indented<'_, '_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for line in text.split_inclusive('\n') {
            if self.line_start {
                for _ in 0..self.depth {
                    self.f.write_str("    ")?;
                }
            }

            self.line_start = line.ends_with('\n');
            self.f.write_str(line)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, Name, NamedDeBruijn, Term, Type, Unique},
        builtins::DefaultFunction,
        machine::{Context, runtime::BuiltinRuntime, value::Value},
    };
    use std::rc::Rc;

    /// Deep enough that a formatter taking one stack frame per level cannot survive the test
    /// harness's ~2 MiB thread stack.
    const DEEP: usize = 100_000;

    // The expected texts were written by the derived `Debug` impls these replace.

    #[test]
    fn terms_print_as_derived() {
        let x = Name {
            text: "x".to_string(),
            unique: Unique::new(0),
        };
        let term: Term<Name> = Term::Apply {
            function: Rc::new(Term::Lambda {
                parameter_name: Rc::new(x.clone()),
                body: Rc::new(Term::Var {
                    name: Rc::new(x),
                    uniq_id: 2,
                }),
                uniq_id: 1,
            }),
            argument: Rc::new(Term::Constr {
                tag: 1,
                fields: vec![
                    Term::Constant {
                        value: Rc::new(Constant::ProtoPair(
                            Type::List(Rc::new(Type::Integer)),
                            Type::Array(Rc::new(Type::Bool)),
                            Rc::new(Constant::ProtoList(
                                Type::Integer,
                                vec![Rc::new(Constant::Integer(7.into()))],
                            )),
                            Rc::new(Constant::ProtoArray(Type::Bool, vec![])),
                        )),
                        uniq_id: 3,
                    },
                    Term::Case {
                        constr: Rc::new(Term::Error { uniq_id: 4 }),
                        branches: vec![],
                        uniq_id: 5,
                    },
                ],
                uniq_id: 6,
            }),
            uniq_id: 7,
        };

        assert_eq!(
            format!("{term:?}"),
            r#"Apply { function: Lambda { parameter_name: Name { text: "x", unique: Unique(0) }, body: Var { name: Name { text: "x", unique: Unique(0) }, uniq_id: 2 }, uniq_id: 1 }, argument: Constr { tag: 1, fields: [Constant { value: ProtoPair(List(Integer), Array(Bool), ProtoList(Integer, [Integer(7)]), ProtoArray(Bool, [])), uniq_id: 3 }, Case { constr: Error { uniq_id: 4 }, branches: [], uniq_id: 5 }], uniq_id: 6 }, uniq_id: 7 }"#
        );
        assert_eq!(
            format!("{term:#?}"),
            r#"Apply {
    function: Lambda {
        parameter_name: Name {
            text: "x",
            unique: Unique(
                0,
            ),
        },
        body: Var {
            name: Name {
                text: "x",
                unique: Unique(
                    0,
                ),
            },
            uniq_id: 2,
        },
        uniq_id: 1,
    },
    argument: Constr {
        tag: 1,
        fields: [
            Constant {
                value: ProtoPair(
                    List(
                        Integer,
                    ),
                    Array(
                        Bool,
                    ),
                    ProtoList(
                        Integer,
                        [
                            Integer(
                                7,
                            ),
                        ],
                    ),
                    ProtoArray(
                        Bool,
                        [],
                    ),
                ),
                uniq_id: 3,
            },
            Case {
                constr: Error {
                    uniq_id: 4,
                },
                branches: [],
                uniq_id: 5,
            },
        ],
        uniq_id: 6,
    },
    uniq_id: 7,
}"#
        );
    }

    #[test]
    fn machine_values_and_frames_print_as_derived() {
        let body: Rc<Term<NamedDeBruijn>> = Rc::new(Term::Builtin {
            fun: DefaultFunction::AddInteger,
            uniq_id: 8,
        });
        let value = Value::Constr {
            tag: 0,
            fields: vec![
                Value::Delay {
                    body: Rc::clone(&body),
                    env: Rc::new(vec![Value::Con(Rc::new(Constant::Unit))]),
                    term_id: 9,
                },
                Value::Builtin {
                    fun: DefaultFunction::IfThenElse,
                    runtime: BuiltinRuntime {
                        args: vec![Value::Con(Rc::new(Constant::Bool(true)))],
                        fun: DefaultFunction::IfThenElse,
                        forces: 1,
                    },
                    term_id: 10,
                },
            ],
            term_id: 11,
        };

        assert_eq!(
            format!("{value:?}"),
            r#"Constr { tag: 0, fields: [Delay { body: Builtin { fun: AddInteger, uniq_id: 8 }, env: [Con(Unit)], term_id: 9 }, Builtin { fun: IfThenElse, runtime: BuiltinRuntime { args: [Con(Bool(true))], fun: IfThenElse, forces: 1 }, term_id: 10 }], term_id: 11 }"#
        );
        assert_eq!(
            format!("{value:#?}"),
            r#"Constr {
    tag: 0,
    fields: [
        Delay {
            body: Builtin {
                fun: AddInteger,
                uniq_id: 8,
            },
            env: [
                Con(
                    Unit,
                ),
            ],
            term_id: 9,
        },
        Builtin {
            fun: IfThenElse,
            runtime: BuiltinRuntime {
                args: [
                    Con(
                        Bool(
                            true,
                        ),
                    ),
                ],
                fun: IfThenElse,
                forces: 1,
            },
            term_id: 10,
        },
    ],
    term_id: 11,
}"#
        );

        let context = Context::FrameForce(Box::new(Context::FrameAwaitArg(
            Value::Con(Rc::new(Constant::Integer(1.into()))),
            Box::new(Context::NoFrame),
        )));

        assert_eq!(
            format!("{context:?}"),
            r#"FrameForce(FrameAwaitArg(Con(Integer(1)), NoFrame))"#
        );
        assert_eq!(
            format!("{context:#?}"),
            r#"FrameForce(
    FrameAwaitArg(
        Con(
            Integer(
                1,
            ),
        ),
        NoFrame,
    ),
)"#
        );
    }

    #[test]
    fn deep_structures_print_without_overflowing_the_stack() {
        let mut term: Term<Name> = Term::Error { uniq_id: 0 };
        let mut constant = Constant::Unit;
        let mut typ = Type::Unit;
        let mut value = Value::Con(Rc::new(Constant::Unit));
        let mut context = Context::NoFrame;

        for level in 0..DEEP {
            term = if level % 2 == 0 {
                Term::Delay {
                    body: Rc::new(term),
                    uniq_id: 0,
                }
            } else {
                Term::Constr {
                    tag: 0,
                    fields: vec![term],
                    uniq_id: 0,
                }
            };
            constant = Constant::ProtoList(Type::Unit, vec![Rc::new(constant)]);
            typ = Type::List(Rc::new(typ));
            value = if level % 2 == 0 {
                Value::Constr {
                    tag: 0,
                    fields: vec![value],
                    term_id: 0,
                }
            } else {
                Value::Delay {
                    body: Rc::new(Term::Error { uniq_id: 0 }),
                    env: Rc::new(vec![value]),
                    term_id: 0,
                }
            };
            context = Context::FrameForce(Box::new(context));
        }

        assert!(format!("{term:?}").starts_with("Constr { tag: 0, fields: [Delay { body: Constr {"));
        assert!(format!("{constant:?}").starts_with("ProtoList(Unit, [ProtoList(Unit, ["));
        assert!(format!("{typ:?}").ends_with(&format!("List(Unit{}", ")".repeat(DEEP))));
        assert!(format!("{value:?}").starts_with("Delay { body: Error { uniq_id: 0 }, env: [Constr {"));
        assert!(format!("{context:?}").ends_with(&format!("FrameForce(NoFrame{}", ")".repeat(DEEP))));
    }
}
