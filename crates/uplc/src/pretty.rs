//! Textual UPLC.
//!
//! Terms, constants, types and Data nest as deep as the script makes them (see the `Drop` impl on
//! `Term` in `ast.rs`), so the documents are built bottom-up off explicit stacks, and they live in
//! an arena: a document tree held by `Rc`s would be freed one stack frame per level, while the
//! arena frees its nodes side by side. The layout itself (`pretty`'s renderer) already works off
//! a stack of its own. The documents are the same ones a recursive builder produced, node for
//! node, so the text is too.
//!
//! The text is at least as large as the script is deep: each nested `delay`, `lam` or `[` that
//! does not fit on a line starts a line indented two columns further than the last, so a term
//! `n` levels deep prints as roughly `n * n` characters.

use crate::{
    ast::{Constant, Program, Term, Type},
    flat::Binder,
    machine::{
        runtime::{Compressable, convert_tag_to_constr},
        value::from_pallas_bigint,
    },
};
use pallas_primitives::conway::{Constr, PlutusData};
use pretty::{Arena, DocAllocator, DocBuilder, RefDoc};
use std::ascii::escape_default;

type Doc<'d> = RefDoc<'d, ()>;
type Builder<'d> = DocBuilder<'d, Arena<'d, ()>, ()>;

/// Lay `doc` out on 80 columns.
fn render(doc: Builder<'_>) -> String {
    let mut w = Vec::new();

    doc.into_doc().render(80, &mut w).unwrap();

    String::from_utf8(w)
        .unwrap()
        .lines()
        // This is a hack to deal with blank newlines
        // that end up with a bunch of useless whitespace
        // because of the nesting
        .map(|l| {
            if l.chars().all(|c| c.is_whitespace()) {
                "".to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl<'a, T> Program<T>
where
    T: Binder<'a>,
{
    pub fn to_pretty(&self) -> String {
        let arena = Arena::new();
        let version = format!("{}.{}.{}", self.version.0, self.version.1, self.version.2);

        let doc = arena
            .text("(")
            .append(arena.text("program"))
            .append(arena.line())
            .append(arena.text(version))
            .append(arena.line())
            .append(term_doc(&arena, &self.term))
            .nest(2)
            .append(arena.line_())
            .append(arena.text(")"));

        render(doc)
    }
}

impl<'a, T> Term<T>
where
    T: Binder<'a>,
{
    pub fn to_pretty(&self) -> String {
        let arena = Arena::new();

        render(arena.nil().append(term_doc(&arena, self)))
    }
}

impl Constant {
    pub fn to_pretty(&self) -> String {
        let arena = Arena::new();

        render(arena.nil().append(constant_doc(&arena, self)))
    }
}

/// The document of `term`. `done` holds the documents of finished subterms whose parent is still
/// waiting on `tasks`; a parent takes its children's documents back off the end, in order.
fn term_doc<'d, 'a, T: Binder<'a>>(arena: &'d Arena<'d, ()>, term: &Term<T>) -> Doc<'d> {
    enum Task<'t, T> {
        Visit(&'t Term<T>),
        Build(&'t Term<T>),
    }

    let mut tasks = vec![Task::Visit(term)];
    let mut done: Vec<Doc<'d>> = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(term) => {
                tasks.push(Task::Build(term));

                // Children go on in reverse, so their documents come out in order.
                match term {
                    Term::Delay { body, .. }
                    | Term::Lambda { body, .. }
                    | Term::Force { body, .. } => tasks.push(Task::Visit(body)),
                    Term::Apply {
                        function, argument, ..
                    } => {
                        tasks.push(Task::Visit(argument));
                        tasks.push(Task::Visit(function));
                    }
                    Term::Constr { fields, .. } => {
                        tasks.extend(fields.iter().rev().map(Task::Visit));
                    }
                    Term::Case {
                        constr, branches, ..
                    } => {
                        tasks.extend(branches.iter().rev().map(Task::Visit));
                        tasks.push(Task::Visit(constr));
                    }
                    Term::Var { .. }
                    | Term::Constant { .. }
                    | Term::Error { .. }
                    | Term::Builtin { .. } => {}
                }
            }
            Task::Build(term) => {
                let doc = match term {
                    Term::Var { name, .. } => arena.text(name.text()),
                    Term::Delay { .. } => {
                        let body = done.pop().expect("body laid out");

                        arena
                            .text("(")
                            .append(
                                arena
                                    .text("delay")
                                    .append(arena.line())
                                    .append(body)
                                    .nest(2),
                            )
                            .append(arena.line_())
                            .append(arena.text(")"))
                    }
                    Term::Lambda { parameter_name, .. } => {
                        let body = done.pop().expect("body laid out");

                        arena
                            .text("(")
                            .append(
                                arena
                                    .text("lam")
                                    .append(arena.line())
                                    .append(arena.text(parameter_name.text()))
                                    .append(arena.line())
                                    .append(body)
                                    .nest(2),
                            )
                            .append(arena.line_())
                            .append(arena.text(")"))
                    }
                    Term::Apply { .. } => {
                        let argument = done.pop().expect("argument laid out");
                        let function = done.pop().expect("function laid out");

                        arena
                            .text("[")
                            .append(
                                arena
                                    .line()
                                    .append(
                                        arena
                                            .nil()
                                            .append(function)
                                            .append(arena.line())
                                            .append(argument)
                                            .group(),
                                    )
                                    .nest(2),
                            )
                            .append(arena.line())
                            .append(arena.text("]"))
                    }
                    Term::Constant { value, .. } => arena
                        .text("(")
                        .append(
                            arena
                                .text("con")
                                .append(arena.line())
                                .append(constant_doc(arena, value))
                                .nest(2),
                        )
                        .append(arena.line_())
                        .append(arena.text(")")),
                    Term::Force { .. } => {
                        let body = done.pop().expect("body laid out");

                        arena
                            .text("(")
                            .append(
                                arena
                                    .text("force")
                                    .append(arena.line())
                                    .append(body)
                                    .nest(2),
                            )
                            .append(arena.line_())
                            .append(arena.text(")"))
                    }
                    Term::Error { .. } => arena.text("(error)").nest(2),
                    Term::Builtin { fun, .. } => arena
                        .text("(")
                        .append(
                            arena
                                .text("builtin")
                                .append(arena.line())
                                .append(arena.text(fun.to_string()))
                                .nest(2),
                        )
                        .append(arena.line_())
                        .append(arena.text(")")),
                    Term::Constr { tag, fields, .. } => {
                        let fields = done.split_off(done.len() - fields.len());

                        arena
                            .text("(")
                            .append(
                                arena
                                    .text("constr")
                                    .append(arena.line())
                                    .append(arena.as_string(tag))
                                    .nest(2),
                            )
                            .append(if fields.is_empty() {
                                arena.nil()
                            } else {
                                arena
                                    .line()
                                    .append(arena.intersperse(fields, arena.line()).nest(2))
                                    .append(arena.line_())
                            })
                            .append(arena.text(")"))
                    }
                    Term::Case { branches, .. } => {
                        let branches = done.split_off(done.len() - branches.len());
                        let constr = done.pop().expect("scrutinee laid out");

                        arena
                            .text("(")
                            .append(
                                arena
                                    .text("case")
                                    .append(arena.line())
                                    .append(constr)
                                    .nest(2),
                            )
                            .append(if branches.is_empty() {
                                arena.nil()
                            } else {
                                arena
                                    .line()
                                    .append(arena.intersperse(branches, arena.line()).nest(2))
                                    .append(arena.line_())
                            })
                            .append(arena.text(")"))
                    }
                };

                done.push(doc.group().into_doc());
            }
        }
    }

    done.pop().expect("the document of the root")
}

/// The text of a string constant, escaped.
fn escaped(s: &str) -> String {
    String::from_utf8(
        s.as_bytes()
            .iter()
            .flat_map(|c| escape_default(*c).collect::<Vec<u8>>())
            .collect(),
    )
    .unwrap()
}

/// `constant` with its type: `integer 42`, `(list integer) [1, 2]`.
fn constant_doc<'d>(arena: &'d Arena<'d, ()>, constant: &Constant) -> Doc<'d> {
    let doc = match constant {
        Constant::Integer(i) => arena
            .text("integer")
            .append(arena.line())
            .append(arena.as_string(i)),
        Constant::ByteString(bs) => arena
            .text("bytestring")
            .append(arena.line())
            .append(arena.text("#"))
            .append(arena.text(hex::encode(bs))),
        Constant::String(s) => arena
            .text("string")
            .append(arena.line())
            .append(arena.text("\""))
            .append(arena.text(escaped(s)))
            .append(arena.text("\"")),
        Constant::Unit => arena
            .text("unit")
            .append(arena.line())
            .append(arena.text("()")),
        Constant::Bool(b) => arena
            .text("bool")
            .append(arena.line())
            .append(arena.text(if *b { "True" } else { "False" })),
        Constant::ProtoList(r#type, items) => arena
            .text("(")
            .append("list")
            .append(arena.space())
            .append(type_doc(arena, r#type))
            .append(")")
            .append(arena.line())
            .append(arena.text("["))
            .append(arena.intersperse(
                items.iter().map(|c| list_item_doc(arena, c)),
                arena.text(", "),
            ))
            .append(arena.text("]")),
        Constant::ProtoPair(type_left, type_right, left, right) => arena
            .text("(")
            .append("pair")
            .append(arena.space())
            .append(type_doc(arena, type_left))
            .append(arena.space())
            .append(type_doc(arena, type_right))
            .append(arena.text(")"))
            .append(arena.line())
            .append(arena.text("("))
            .append(list_item_doc(arena, left))
            .append(arena.text(", "))
            .append(list_item_doc(arena, right))
            .append(arena.text(")")),
        Constant::Data(d) => arena
            .text("data ")
            .append(arena.text("("))
            .append(data_doc(arena, d))
            .append(arena.text(")")),
        Constant::Bls12_381G1Element(p1) => arena
            .text("bls12_381_G1_element ")
            .append(arena.line())
            .append(arena.text("0x"))
            .append(arena.text(hex::encode(p1.compress()))),
        Constant::Bls12_381G2Element(p2) => arena
            .text("bls12_381_G2_element ")
            .append(arena.line())
            .append(arena.text("0x"))
            .append(arena.text(hex::encode(p2.compress()))),
        Constant::Bls12_381MlResult(_) => panic!("cannot represent Bls12_381MlResult as text"),
        Constant::ProtoArray(r#type, items) => arena
            .text("(")
            .append("array")
            .append(arena.space())
            .append(type_doc(arena, r#type))
            .append(")")
            .append(arena.line())
            .append(arena.text("["))
            .append(arena.intersperse(
                items.iter().map(|c| list_item_doc(arena, c)),
                arena.text(", "),
            ))
            .append(arena.text("]")),
        Constant::Value(value) => arena
            .text("value")
            .append(arena.line())
            .append(value_doc(arena, value)),
    };

    doc.into_doc()
}

/// `constant` without its type, as it appears inside a list, an array or a pair. Lists, arrays
/// and pairs of them nest as deep as their type, so they are laid out bottom-up: `done` holds the
/// documents of finished elements whose container is still waiting on `tasks`.
fn list_item_doc<'d>(arena: &'d Arena<'d, ()>, constant: &Constant) -> Doc<'d> {
    enum Task<'c> {
        Visit(&'c Constant),
        Build(&'c Constant),
    }

    let mut tasks = vec![Task::Visit(constant)];
    let mut done: Vec<Doc<'d>> = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(constant) => match constant {
                Constant::ProtoList(_, items) | Constant::ProtoArray(_, items) => {
                    tasks.push(Task::Build(constant));
                    tasks.extend(items.iter().rev().map(|item| Task::Visit(item)));
                }
                Constant::ProtoPair(_, _, left, right) => {
                    tasks.push(Task::Build(constant));
                    tasks.push(Task::Visit(right));
                    tasks.push(Task::Visit(left));
                }
                other => done.push(scalar_item_doc(arena, other)),
            },
            Task::Build(constant) => {
                let doc = match constant {
                    Constant::ProtoList(_, items) | Constant::ProtoArray(_, items) => {
                        let items = done.split_off(done.len() - items.len());

                        arena
                            .text("[")
                            .append(arena.intersperse(items, arena.text(", ")))
                            .append(arena.text("]"))
                    }
                    Constant::ProtoPair(..) => {
                        let right = done.pop().expect("second component laid out");
                        let left = done.pop().expect("first component laid out");

                        arena
                            .text("(")
                            .append(left)
                            .append(arena.text(", "))
                            .append(right)
                            .append(arena.text(")"))
                    }
                    _ => unreachable!("only lists, arrays and pairs are built"),
                };

                done.push(doc.into_doc());
            }
        }
    }

    done.pop().expect("the document of the root")
}

/// A constant that is neither a list, an array nor a pair, without its type.
fn scalar_item_doc<'d>(arena: &'d Arena<'d, ()>, constant: &Constant) -> Doc<'d> {
    let doc = match constant {
        Constant::Integer(i) => arena.as_string(i),
        Constant::ByteString(bs) => arena.text("#").append(arena.text(hex::encode(bs))),
        Constant::String(s) => arena
            .text("\"")
            .append(arena.text(escaped(s)))
            .append(arena.text("\"")),
        Constant::Unit => arena.text("()"),
        Constant::Bool(b) => arena.text(if *b { "True" } else { "False" }),
        Constant::Data(data) => arena.nil().append(data_doc(arena, data)),
        Constant::Bls12_381G1Element(p1) => arena
            .text("0x")
            .append(arena.text(hex::encode(p1.compress()))),
        Constant::Bls12_381G2Element(p2) => arena
            .text("0x")
            .append(arena.text(hex::encode(p2.compress()))),
        Constant::Bls12_381MlResult(_) => panic!("cannot represent Bls12_381MlResult as text"),
        Constant::Value(value) => arena.nil().append(value_doc(arena, value)),
        Constant::ProtoList(..) | Constant::ProtoArray(..) | Constant::ProtoPair(..) => {
            unreachable!("lists, arrays and pairs are laid out by `list_item_doc`")
        }
    };

    doc.into_doc()
}

fn value_doc<'d>(arena: &'d Arena<'d, ()>, value: &crate::ast::Value) -> Doc<'d> {
    arena
        .text("[")
        .append(arena.intersperse(
            value.iter().map(|(currency, inner)| {
                arena
                    .text("(")
                    .append(arena.text("#"))
                    .append(arena.text(hex::encode(currency)))
                    .append(arena.text(", "))
                    .append(arena.text("["))
                    .append(arena.intersperse(
                        inner.map(|(token, quantity)| {
                            arena
                                .text("(")
                                .append(arena.text("#"))
                                .append(arena.text(hex::encode(token)))
                                .append(arena.text(", "))
                                .append(arena.as_string(quantity))
                                .append(arena.text(")"))
                        }),
                        arena.text(", "),
                    ))
                    .append(arena.text("]"))
                    .append(arena.text(")"))
            }),
            arena.text(", "),
        ))
        .append(arena.text("]"))
        .into_doc()
}

/// Data, laid out bottom-up: `done` holds the documents of finished values whose container is
/// still waiting on `tasks`.
fn data_doc<'d>(arena: &'d Arena<'d, ()>, data: &PlutusData) -> Doc<'d> {
    enum Task<'p> {
        Visit(&'p PlutusData),
        Build(&'p PlutusData),
    }

    let mut tasks = vec![Task::Visit(data)];
    let mut done: Vec<Doc<'d>> = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(data) => {
                tasks.push(Task::Build(data));

                match data {
                    PlutusData::Constr(Constr { fields, .. }) => {
                        tasks.extend(fields.iter().rev().map(Task::Visit));
                    }
                    PlutusData::Map(kvp) => {
                        for (key, value) in kvp.iter().rev() {
                            tasks.push(Task::Visit(value));
                            tasks.push(Task::Visit(key));
                        }
                    }
                    PlutusData::Array(a) => tasks.extend(a.iter().rev().map(Task::Visit)),
                    PlutusData::BigInt(_) | PlutusData::BoundedBytes(_) => {}
                }
            }
            Task::Build(data) => {
                let doc = match data {
                    PlutusData::Constr(Constr {
                        tag,
                        any_constructor,
                        fields,
                    }) => {
                        let fields = done.split_off(done.len() - fields.len());

                        arena
                            .text("Constr")
                            .append(arena.space())
                            .append(
                                arena.as_string(
                                    convert_tag_to_constr(*tag)
                                        .unwrap_or_else(|| any_constructor.unwrap()),
                                ),
                            )
                            .append(arena.space())
                            .append(arena.text("["))
                            .append(arena.intersperse(fields, arena.text(", ")))
                            .append(arena.text("]"))
                    }
                    PlutusData::Map(kvp) => {
                        let mut entries = done.split_off(done.len() - 2 * kvp.len()).into_iter();
                        let mut pairs = Vec::with_capacity(kvp.len());

                        while let (Some(key), Some(value)) = (entries.next(), entries.next()) {
                            pairs.push(
                                arena
                                    .text("(")
                                    .append(key)
                                    .append(arena.text(", "))
                                    .append(value)
                                    .append(arena.text(")")),
                            );
                        }

                        arena
                            .text("Map")
                            .append(arena.space())
                            .append(arena.text("["))
                            .append(arena.intersperse(pairs, arena.text(", ")))
                            .append(arena.text("]"))
                    }
                    PlutusData::BigInt(bi) => arena
                        .text("I")
                        .append(arena.space())
                        .append(arena.text(from_pallas_bigint(bi).to_string())),
                    PlutusData::BoundedBytes(bs) => arena
                        .text("B")
                        .append(arena.space())
                        .append(arena.text("#"))
                        .append(arena.text(hex::encode(bs.to_vec()))),
                    PlutusData::Array(a) => {
                        let items = done.split_off(done.len() - a.len());

                        arena
                            .text("List")
                            .append(arena.space())
                            .append(arena.text("["))
                            .append(arena.intersperse(items, arena.text(", ")))
                            .append(arena.text("]"))
                    }
                };

                done.push(doc.into_doc());
            }
        }
    }

    done.pop().expect("the document of the root")
}

/// `typ`, laid out bottom-up: `done` holds the documents of finished operands whose type is still
/// waiting on `tasks`.
fn type_doc<'d>(arena: &'d Arena<'d, ()>, typ: &Type) -> Doc<'d> {
    enum Task<'t> {
        Visit(&'t Type),
        Build(&'t Type),
    }

    let mut tasks = vec![Task::Visit(typ)];
    let mut done: Vec<Doc<'d>> = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(typ) => match typ {
                Type::List(inner) | Type::Array(inner) => {
                    tasks.push(Task::Build(typ));
                    tasks.push(Task::Visit(inner));
                }
                Type::Pair(l, r) => {
                    tasks.push(Task::Build(typ));
                    tasks.push(Task::Visit(r));
                    tasks.push(Task::Visit(l));
                }
                Type::Bool => done.push(arena.text("bool").into_doc()),
                Type::Integer => done.push(arena.text("integer").into_doc()),
                Type::String => done.push(arena.text("string").into_doc()),
                Type::ByteString => done.push(arena.text("bytestring").into_doc()),
                Type::Unit => done.push(arena.text("unit").into_doc()),
                Type::Data => done.push(arena.text("data").into_doc()),
                Type::Bls12_381G1Element => {
                    done.push(arena.text("bls12_381_G1_element").into_doc())
                }
                Type::Bls12_381G2Element => {
                    done.push(arena.text("bls12_381_G2_element").into_doc())
                }
                Type::Bls12_381MlResult => done.push(arena.text("bls12_381_mlresult").into_doc()),
                Type::Value => done.push(arena.text("value").into_doc()),
            },
            Task::Build(typ) => {
                let doc = match typ {
                    Type::List(_) => {
                        let inner = done.pop().expect("operand laid out");

                        arena
                            .text("(list")
                            .append(arena.line())
                            .append(inner)
                            .append(arena.line_())
                            .append(")")
                    }
                    Type::Pair(_, _) => {
                        let r = done.pop().expect("second operand laid out");
                        let l = done.pop().expect("first operand laid out");

                        arena
                            .text("(pair")
                            .append(arena.line())
                            .append(l)
                            .append(arena.line())
                            .append(r)
                            .append(")")
                    }
                    Type::Array(_) => {
                        let inner = done.pop().expect("operand laid out");

                        arena
                            .text("(array")
                            .append(arena.line())
                            .append(inner)
                            .append(arena.line_())
                            .append(")")
                    }
                    _ => unreachable!("only lists, arrays and pairs are built"),
                };

                done.push(doc.into_doc());
            }
        }
    }

    done.pop().expect("the document of the root")
}

#[cfg(test)]
mod tests {
    use crate::ast::{Constant, Name, Term, Type};
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use std::rc::Rc;

    #[test]
    fn format_arrays() {
        let uplc = "(program 1.1.0 (con (list (array (pair bool bls12_381_G2_element))) [[], []]))";

        assert_eq!(
            crate::parser::program(uplc).unwrap().to_pretty(),
            indoc! {
                r#"
                (program
                  1.1.0
                  (con (list (array (pair bool bls12_381_G2_element))) [[], []])
                )"#
            }
        );

        let uplc = "(program 1.1.0 (con (array (list integer)) [[1, 2], []]))";
        let pretty = crate::parser::program(uplc).unwrap().to_pretty();

        assert_eq!(
            pretty,
            indoc! {
                r#"
                (program
                  1.1.0
                  (con (array (list integer)) [[1, 2], []])
                )"#
            }
        );

        assert_eq!(
            crate::parser::program(&pretty).unwrap(),
            crate::parser::program(uplc).unwrap()
        );
    }

    /// Lays out `f()` on a thread whose stack is far too small for a recursive layout of the
    /// inputs below.
    fn on_small_stack(f: impl FnOnce() -> String + Send + 'static) -> String {
        std::thread::Builder::new()
            .stack_size(128 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap()
    }

    #[test]
    fn deep_constants_and_types_print_on_a_small_stack() {
        // Types and constant values do not indent per level, so their text stays linear in depth.
        const DEEP: usize = 50_000;

        let text = on_small_stack(|| {
            let mut constant = Constant::Integer(7.into());
            let mut typ = Type::Integer;

            for _ in 0..DEEP {
                constant = Constant::ProtoList(typ.clone(), vec![Rc::new(constant)]);
                typ = Type::List(Rc::new(typ));
            }

            constant.to_pretty()
        });

        assert!(text.starts_with(&format!("(list {}integer", "(list\n".repeat(DEEP - 1))));
        assert!(text.ends_with(&format!("{}7{}", "[".repeat(DEEP), "]".repeat(DEEP))));
    }

    #[test]
    fn deep_terms_print_on_a_small_stack() {
        // Each level of a term indents its body further, so the text grows with the square of the
        // depth; this depth is far beyond what a recursive layout managed on this stack.
        const DEPTH: usize = 2_000;

        let text = on_small_stack(|| {
            let mut term: Term<Name> = Term::Error { uniq_id: 0 };

            for level in 0..DEPTH {
                term = if level % 2 == 0 {
                    Term::Delay {
                        body: Rc::new(term),
                        uniq_id: 0,
                    }
                } else {
                    Term::Constr {
                        tag: 1,
                        fields: vec![term],
                        uniq_id: 0,
                    }
                };
            }

            term.to_pretty()
        });

        assert!(text.starts_with("(constr\n  1\n(delay\n    (constr\n      1\n    (delay\n"));
        assert_eq!(text.lines().count(), 5 * DEPTH / 2 + 1);
    }

    #[test]
    fn format_pair_bool_pair_integer_bytestring() {
        let uplc = "(program 0.0.0 (con (pair bool (pair integer bytestring)) (True, (14, #42))))";

        assert_eq!(
            crate::parser::program(uplc).unwrap().to_pretty(),
            indoc! {
                r#"
                (program
                  0.0.0
                  (con (pair bool (pair integer bytestring)) (True, (14, #42)))
                )"#
            }
        )
    }
}
