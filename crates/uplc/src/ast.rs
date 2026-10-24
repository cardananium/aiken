use crate::{
    builtins::DefaultFunction,
    debruijn::{self, Converter},
    debug::{self, Item},
    flat::Binder,
    machine::{
        Machine,
        cost_model::{
            CostModel, ExBudget, initialize_cost_model, initialize_cost_model_with_protocol,
        },
        eval_result::EvalResult,
        value::to_pallas_bigint,
    },
    optimize::interner::CodeGenInterner,
    tx::script_context::PlutusScript,
};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use pallas_addresses::{Network, ShelleyAddress, ShelleyDelegationPart, ShelleyPaymentPart};
use pallas_primitives::{
    alonzo::{Constr, PlutusData},
    conway::{self, Language},
};
use pallas_traverse::ComputeHash;
use serde::{
    self,
    de::{self, Deserialize, Deserializer, MapAccess, Visitor},
    ser::{Serialize, SerializeStruct, Serializer},
};
use std::{
    collections::{BTreeMap, HashSet},
    convert::AsRef,
    fmt::{self, Display},
    hash::{self, Hash},
    mem,
    rc::Rc,
};

use crate::global_uniq::next_uniq_id;

/// This represents a program in Untyped Plutus Core.
/// A program contains a version tuple and a term.
/// It is generic because Term requires a generic type.
#[derive(Debug, Clone, PartialEq)]
pub struct Program<T> {
    pub version: (usize, usize, usize),
    pub term: Term<T>,
}

impl<T> Program<T>
where
    T: Clone,
{
    /// We use this to apply the validator to Datum,
    /// then redeemer, then ScriptContext. If datum is
    /// even necessary (i.e. minting policy).
    pub fn apply(&self, program: &Self) -> Self {
        let applied_term = Term::Apply {
            function: Rc::new(self.term.clone()),
            argument: Rc::new(program.term.clone()),
            uniq_id: next_uniq_id(),
        };

        Program {
            version: self.version,
            term: applied_term,
        }
    }

    /// A convenient and faster version that `apply_term` since the program doesn't need to be
    /// re-interned (constant Data do not introduce new bindings).
    pub fn apply_data(&self, plutus_data: PlutusData) -> Self {
        let applied_term = Term::Apply {
            function: Rc::new(self.term.clone()),
            argument: Rc::new(Term::Constant {
                value: Constant::Data(plutus_data).into(),
                uniq_id: next_uniq_id(),
            }),
            uniq_id: next_uniq_id(),
        };

        Program {
            version: self.version,
            term: applied_term,
        }
    }
}

impl Program<Name> {
    /// We use this to apply the validator to Datum,
    /// then redeemer, then ScriptContext. If datum is
    /// even necessary (i.e. minting policy).
    pub fn apply_term(&self, term: &Term<Name>) -> Self {
        let applied_term = Term::Apply {
            function: Rc::new(self.term.clone()),
            argument: Rc::new(term.clone()),
            uniq_id: next_uniq_id(),
        };

        let mut program = Program {
            version: self.version,
            term: applied_term,
        };

        CodeGenInterner::new().program(&mut program);

        program
    }

    /// A convenient method to convery named programs to debruijn programs.
    pub fn to_debruijn(self) -> Result<Program<DeBruijn>, debruijn::Error> {
        self.try_into()
    }

    /// A convenient method to convery named programs to named debruijn programs.
    pub fn to_named_debruijn(self) -> Result<Program<NamedDeBruijn>, debruijn::Error> {
        self.try_into()
    }
}

impl<'a, T> Display for Program<T>
where
    T: Binder<'a>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_pretty())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SerializableProgram {
    PlutusV1Program(Program<DeBruijn>),
    PlutusV2Program(Program<DeBruijn>),
    PlutusV3Program(Program<DeBruijn>),
}

impl SerializableProgram {
    pub fn inner(&self) -> &Program<DeBruijn> {
        use SerializableProgram::*;

        match self {
            PlutusV1Program(program) => program,
            PlutusV2Program(program) => program,
            PlutusV3Program(program) => program,
        }
    }

    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(Program<DeBruijn>) -> Program<DeBruijn>,
    {
        use SerializableProgram::*;

        match self {
            PlutusV1Program(program) => PlutusV1Program(f(program)),
            PlutusV2Program(program) => PlutusV2Program(f(program)),
            PlutusV3Program(program) => PlutusV3Program(f(program)),
        }
    }

    pub fn compiled_code_and_hash(&self) -> (pallas_crypto::hash::Hash<28>, PlutusScript) {
        use SerializableProgram::*;

        match self {
            PlutusV1Program(pgrm) => {
                let cbor = pgrm.to_cbor().unwrap();
                let script = conway::PlutusScript::<1>(cbor.into());
                let hash = script.compute_hash();
                (hash, PlutusScript::V1(script))
            }

            PlutusV2Program(pgrm) => {
                let cbor = pgrm.to_cbor().unwrap();
                let script = conway::PlutusScript::<2>(cbor.into());
                let hash = script.compute_hash();
                (hash, PlutusScript::V2(script))
            }

            PlutusV3Program(pgrm) => {
                let cbor = pgrm.to_cbor().unwrap();
                let script = conway::PlutusScript::<3>(cbor.into());
                let hash = script.compute_hash();
                (hash, PlutusScript::V3(script))
            }
        }
    }
}

impl Serialize for SerializableProgram {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (hash, compiled_code) = self.compiled_code_and_hash();
        let mut s = serializer.serialize_struct("Program<DeBruijn>", 2)?;
        s.serialize_field("compiledCode", &hex::encode(compiled_code.as_ref()))?;
        s.serialize_field("hash", &hash)?;
        s.end()
    }
}

impl<'a> Deserialize<'a> for SerializableProgram {
    fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "camelCase")]
        enum Fields {
            CompiledCode,
            Hash,
        }

        struct ProgramVisitor;

        impl<'a> Visitor<'a> for ProgramVisitor {
            type Value = SerializableProgram;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("validator")
            }

            fn visit_map<V>(self, mut map: V) -> Result<SerializableProgram, V::Error>
            where
                V: MapAccess<'a>,
            {
                let mut compiled_code: Option<String> = None;
                let mut hash: Option<String> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Fields::CompiledCode => {
                            if compiled_code.is_some() {
                                return Err(de::Error::duplicate_field("compiledCode"));
                            }
                            compiled_code = Some(map.next_value()?);
                        }

                        Fields::Hash => {
                            if hash.is_some() {
                                return Err(de::Error::duplicate_field("hash"));
                            }
                            hash = Some(map.next_value()?);
                        }
                    }
                }
                let compiled_code =
                    compiled_code.ok_or_else(|| de::Error::missing_field("compiledCode"))?;

                let hash = hash.ok_or_else(|| de::Error::missing_field("hash"))?;

                let mut cbor_buffer = Vec::new();
                let mut flat_buffer = Vec::new();

                Program::<DeBruijn>::from_hex(&compiled_code, &mut cbor_buffer, &mut flat_buffer)
                    .map_err(|e| {
                        de::Error::invalid_value(
                            de::Unexpected::Other(&format!("{e}")),
                            &"a base16-encoded CBOR-serialized UPLC program",
                        )
                    })
                    .and_then(|program| {
                        let cbor = || program.to_cbor().unwrap().into();

                        if conway::PlutusScript::<3>(cbor()).compute_hash().to_string() == hash {
                            return Ok(SerializableProgram::PlutusV3Program(program));
                        }

                        if conway::PlutusScript::<2>(cbor()).compute_hash().to_string() == hash {
                            return Ok(SerializableProgram::PlutusV2Program(program));
                        }

                        if conway::PlutusScript::<1>(cbor()).compute_hash().to_string() == hash {
                            return Ok(SerializableProgram::PlutusV1Program(program));
                        }

                        Err(de::Error::custom(
                            "hash doesn't match any recognisable Plutus version.",
                        ))
                    })
            }
        }

        const FIELDS: &[&str] = &["compiledCode", "hash"];
        deserializer.deserialize_struct("Program<DeBruijn>", FIELDS, ProgramVisitor)
    }
}

impl Program<DeBruijn> {
    pub fn address(
        &self,
        network: Network,
        delegation: ShelleyDelegationPart,
        plutus_version: &Language,
    ) -> ShelleyAddress {
        let cbor = self.to_cbor().unwrap();

        let validator_hash = match plutus_version {
            Language::PlutusV1 => conway::PlutusScript::<1>(cbor.into()).compute_hash(),
            Language::PlutusV2 => conway::PlutusScript::<2>(cbor.into()).compute_hash(),
            Language::PlutusV3 => conway::PlutusScript::<3>(cbor.into()).compute_hash(),
        };

        ShelleyAddress::new(
            network,
            ShelleyPaymentPart::Script(validator_hash),
            delegation,
        )
    }
}

/// This represents a term in Untyped Plutus Core.
/// We need a generic type for the different forms that a program may be in.
/// Specifically, `Var` and `parameter_name` in `Lambda` can be a `Name`,
/// `NamedDebruijn`, or `DeBruijn`. When encoded to flat for on chain usage
/// we must encode using the `DeBruijn` form.
pub enum Term<T> {
    // tag: 0
    Var {
        name: Rc<T>,
        uniq_id: isize,
    },
    // tag: 1
    Delay {
        body: Rc<Term<T>>,
        uniq_id: isize,
    },
    // tag: 2
    Lambda {
        parameter_name: Rc<T>,
        body: Rc<Term<T>>,
        uniq_id: isize,
    },
    // tag: 3
    Apply {
        function: Rc<Term<T>>,
        argument: Rc<Term<T>>,
        uniq_id: isize,
    },
    // tag: 4
    Constant {
        value: Rc<Constant>,
        uniq_id: isize,
    },
    // tag: 5
    Force {
        body: Rc<Term<T>>,
        uniq_id: isize,
    },
    // tag: 6
    Error {
        uniq_id: isize,
    },
    // tag: 7
    Builtin {
        fun: DefaultFunction,
        uniq_id: isize,
    },
    // tag: 8
    Constr {
        tag: usize,
        fields: Vec<Term<T>>,
        uniq_id: isize,
    },
    // tag: 9
    Case {
        constr: Rc<Term<T>>,
        branches: Vec<Term<T>>,
        uniq_id: isize,
    },
}

impl<T: Clone> Term<T> {
    /// Rebuild the term with every `Rc` allocation freshly boxed, sharing
    /// nothing with `self`. Unlike `clone`, this only ever borrows existing
    /// allocations (it never bumps a reference count), so it is safe to call
    /// from several threads on terms that share `Rc` nodes — which `Rc`'s
    /// non-atomic counts would otherwise forbid.
    ///
    /// Iterative, like every walk over a term's depth (see the `Drop` impl below): `copies` holds
    /// the finished copies of children whose parent is still waiting on `tasks`.
    pub fn deep_clone(&self) -> Term<T> {
        enum Task<'a, T> {
            Copy(&'a Term<T>),
            /// Rebuild this node around the last copies made, its children.
            Assemble(&'a Term<T>),
        }

        let mut tasks = vec![Task::Copy(self)];
        let mut copies: Vec<Term<T>> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Copy(term) => {
                    tasks.push(Task::Assemble(term));

                    // Children go on in reverse, so their copies come out in order.
                    match term {
                        Term::Delay { body, .. }
                        | Term::Lambda { body, .. }
                        | Term::Force { body, .. } => tasks.push(Task::Copy(body)),
                        Term::Apply {
                            function, argument, ..
                        } => {
                            tasks.push(Task::Copy(argument));
                            tasks.push(Task::Copy(function));
                        }
                        Term::Constr { fields, .. } => {
                            tasks.extend(fields.iter().rev().map(Task::Copy));
                        }
                        Term::Case {
                            constr, branches, ..
                        } => {
                            tasks.extend(branches.iter().rev().map(Task::Copy));
                            tasks.push(Task::Copy(constr));
                        }
                        Term::Var { .. }
                        | Term::Constant { .. }
                        | Term::Error { .. }
                        | Term::Builtin { .. } => {}
                    }
                }
                Task::Assemble(term) => {
                    let copy = match term {
                        Term::Var { name, uniq_id } => Term::Var {
                            name: Rc::new(name.as_ref().clone()),
                            uniq_id: *uniq_id,
                        },
                        Term::Delay { uniq_id, .. } => Term::Delay {
                            body: Rc::new(copies.pop().expect("body copied")),
                            uniq_id: *uniq_id,
                        },
                        Term::Lambda {
                            parameter_name,
                            uniq_id,
                            ..
                        } => Term::Lambda {
                            parameter_name: Rc::new(parameter_name.as_ref().clone()),
                            body: Rc::new(copies.pop().expect("body copied")),
                            uniq_id: *uniq_id,
                        },
                        Term::Apply { uniq_id, .. } => {
                            let argument = Rc::new(copies.pop().expect("argument copied"));
                            let function = Rc::new(copies.pop().expect("function copied"));

                            Term::Apply {
                                function,
                                argument,
                                uniq_id: *uniq_id,
                            }
                        }
                        Term::Constant { value, uniq_id } => Term::Constant {
                            value: Rc::new(value.deep_clone()),
                            uniq_id: *uniq_id,
                        },
                        Term::Force { uniq_id, .. } => Term::Force {
                            body: Rc::new(copies.pop().expect("body copied")),
                            uniq_id: *uniq_id,
                        },
                        Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
                        Term::Builtin { fun, uniq_id } => Term::Builtin {
                            fun: *fun,
                            uniq_id: *uniq_id,
                        },
                        Term::Constr {
                            tag,
                            fields,
                            uniq_id,
                        } => Term::Constr {
                            tag: *tag,
                            fields: copies.split_off(copies.len() - fields.len()),
                            uniq_id: *uniq_id,
                        },
                        Term::Case {
                            branches, uniq_id, ..
                        } => {
                            let branches = copies.split_off(copies.len() - branches.len());
                            let constr = Rc::new(copies.pop().expect("scrutinee copied"));

                            Term::Case {
                                constr,
                                branches,
                                uniq_id: *uniq_id,
                            }
                        }
                    };

                    copies.push(copy);
                }
            }
        }

        copies.pop().expect("the copy of the root")
    }
}

/// Every child of a term sits behind an `Rc` and clones in O(1), except the fields of a `Constr`
/// and the branches of a `Case`, which are owned outright. The derived `Clone` copied those one
/// stack frame per level, and a `constr` can be nested as deep as any other term (see the `Drop`
/// impl below), so they are copied off an explicit stack instead. A term without them -- the
/// machine clones one at nearly every step -- is copied in place, with no allocation.
impl<T> Clone for Term<T> {
    fn clone(&self) -> Self {
        fn shallow<T>(term: &Term<T>) -> Option<Term<T>> {
            Some(match term {
                Term::Var { name, uniq_id } => Term::Var {
                    name: Rc::clone(name),
                    uniq_id: *uniq_id,
                },
                Term::Delay { body, uniq_id } => Term::Delay {
                    body: Rc::clone(body),
                    uniq_id: *uniq_id,
                },
                Term::Lambda {
                    parameter_name,
                    body,
                    uniq_id,
                } => Term::Lambda {
                    parameter_name: Rc::clone(parameter_name),
                    body: Rc::clone(body),
                    uniq_id: *uniq_id,
                },
                Term::Apply {
                    function,
                    argument,
                    uniq_id,
                } => Term::Apply {
                    function: Rc::clone(function),
                    argument: Rc::clone(argument),
                    uniq_id: *uniq_id,
                },
                Term::Constant { value, uniq_id } => Term::Constant {
                    value: Rc::clone(value),
                    uniq_id: *uniq_id,
                },
                Term::Force { body, uniq_id } => Term::Force {
                    body: Rc::clone(body),
                    uniq_id: *uniq_id,
                },
                Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
                Term::Builtin { fun, uniq_id } => Term::Builtin {
                    fun: *fun,
                    uniq_id: *uniq_id,
                },
                Term::Constr { .. } | Term::Case { .. } => return None,
            })
        }

        if let Some(copy) = shallow(self) {
            return copy;
        }

        enum Task<'a, T> {
            Copy(&'a Term<T>),
            /// Rebuild this `constr` or `case` around the last copies made, its children.
            Assemble(&'a Term<T>),
        }

        let mut tasks = vec![Task::Copy(self)];
        let mut copies: Vec<Term<T>> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Copy(term) => match shallow(term) {
                    Some(copy) => copies.push(copy),
                    None => {
                        tasks.push(Task::Assemble(term));

                        let children = match term {
                            Term::Constr { fields, .. } => fields,
                            Term::Case { branches, .. } => branches,
                            _ => unreachable!("only constr and case are not shallow"),
                        };

                        tasks.extend(children.iter().rev().map(Task::Copy));
                    }
                },
                Task::Assemble(term) => {
                    let copy = match term {
                        Term::Constr {
                            tag,
                            fields,
                            uniq_id,
                        } => Term::Constr {
                            tag: *tag,
                            fields: copies.split_off(copies.len() - fields.len()),
                            uniq_id: *uniq_id,
                        },
                        Term::Case {
                            constr,
                            branches,
                            uniq_id,
                        } => Term::Case {
                            constr: Rc::clone(constr),
                            branches: copies.split_off(copies.len() - branches.len()),
                            uniq_id: *uniq_id,
                        },
                        _ => unreachable!("only constr and case are assembled"),
                    };

                    copies.push(copy);
                }
            }
        }

        copies.pop().expect("the copy of the root")
    }
}

/// The derived format, `{:?}` and `{:#?}` alike, written without recursion (see `debug.rs`).
impl<T: fmt::Debug> fmt::Debug for Term<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug::write(self, f, |term| match term {
            Term::Var { name, uniq_id } => Item::Struct(
                "Var",
                vec![("name", Item::Leaf(name)), ("uniq_id", Item::Leaf(uniq_id))],
            ),
            Term::Delay { body, uniq_id } => Item::Struct(
                "Delay",
                vec![("body", Item::Node(body)), ("uniq_id", Item::Leaf(uniq_id))],
            ),
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => Item::Struct(
                "Lambda",
                vec![
                    ("parameter_name", Item::Leaf(parameter_name)),
                    ("body", Item::Node(body)),
                    ("uniq_id", Item::Leaf(uniq_id)),
                ],
            ),
            Term::Apply {
                function,
                argument,
                uniq_id,
            } => Item::Struct(
                "Apply",
                vec![
                    ("function", Item::Node(function)),
                    ("argument", Item::Node(argument)),
                    ("uniq_id", Item::Leaf(uniq_id)),
                ],
            ),
            Term::Constant { value, uniq_id } => Item::Struct(
                "Constant",
                vec![
                    ("value", Item::Leaf(value)),
                    ("uniq_id", Item::Leaf(uniq_id)),
                ],
            ),
            Term::Force { body, uniq_id } => Item::Struct(
                "Force",
                vec![("body", Item::Node(body)), ("uniq_id", Item::Leaf(uniq_id))],
            ),
            Term::Error { uniq_id } => {
                Item::Struct("Error", vec![("uniq_id", Item::Leaf(uniq_id))])
            }
            Term::Builtin { fun, uniq_id } => Item::Struct(
                "Builtin",
                vec![("fun", Item::Leaf(fun)), ("uniq_id", Item::Leaf(uniq_id))],
            ),
            Term::Constr {
                tag,
                fields,
                uniq_id,
            } => Item::Struct(
                "Constr",
                vec![
                    ("tag", Item::Leaf(tag)),
                    (
                        "fields",
                        Item::List(fields.iter().map(Item::Node).collect()),
                    ),
                    ("uniq_id", Item::Leaf(uniq_id)),
                ],
            ),
            Term::Case {
                constr,
                branches,
                uniq_id,
            } => Item::Struct(
                "Case",
                vec![
                    ("constr", Item::Node(constr)),
                    (
                        "branches",
                        Item::List(branches.iter().map(Item::Node).collect()),
                    ),
                    ("uniq_id", Item::Leaf(uniq_id)),
                ],
            ),
        })
    }
}

impl<T> Term<T> {
    /// The term's unique id (every variant carries one).
    pub fn uniq_id(&self) -> isize {
        match self {
            Term::Var { uniq_id, .. }
            | Term::Delay { uniq_id, .. }
            | Term::Lambda { uniq_id, .. }
            | Term::Apply { uniq_id, .. }
            | Term::Constant { uniq_id, .. }
            | Term::Force { uniq_id, .. }
            | Term::Error { uniq_id, .. }
            | Term::Builtin { uniq_id, .. }
            | Term::Constr { uniq_id, .. }
            | Term::Case { uniq_id, .. } => *uniq_id,
        }
    }

    /// A constant, possibly under any number of `delay`s and `force`s.
    pub fn is_constant(&self) -> bool {
        let mut term = self;

        loop {
            match term {
                Term::Constant { .. } => return true,
                Term::Delay { body, .. } | Term::Force { body, .. } => term = body,
                _ => return false,
            }
        }
    }

    pub fn is_true(&self) -> bool {
        matches!(self, Term::Constant { value: c, .. } if c.as_ref() == &Constant::Bool(true))
    }

    pub fn is_false(&self) -> bool {
        matches!(self, Term::Constant { value: c, .. } if c.as_ref() == &Constant::Bool(false))
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, Term::Constant { value: c, .. } if c.as_ref() == &Constant::Unit)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Term::Constant { value: c, .. } if matches!(c.as_ref(), &Constant::Integer(_)))
    }
}

/// Structural equality, ignoring term ids. Iterative, like every walk over a term's depth (see the
/// `Drop` impl below): `pending` holds what is still to compare, in the order a recursive
/// comparison visits it -- left children on top, and an outcome already known but due only after
/// some subterms waiting as a `Check`.
impl<T> PartialEq for Term<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        enum Pending<'a, T> {
            Terms(&'a Term<T>, &'a Term<T>),
            Check(bool),
        }

        let mut pending = vec![Pending::Terms(self, other)];

        while let Some(next) = pending.pop() {
            let pair = match next {
                Pending::Check(outcome) => {
                    if !outcome {
                        return false;
                    }
                    continue;
                }
                Pending::Terms(left, right) => (left, right),
            };

            match pair {
                (Term::Var { name: n1, .. }, Term::Var { name: n2, .. }) => {
                    if n1 != n2 {
                        return false;
                    }
                }
                (Term::Delay { body: b1, .. }, Term::Delay { body: b2, .. })
                | (Term::Force { body: b1, .. }, Term::Force { body: b2, .. }) => {
                    pending.push(Pending::Terms(b1, b2));
                }
                (
                    Term::Lambda {
                        parameter_name: p1,
                        body: b1,
                        ..
                    },
                    Term::Lambda {
                        parameter_name: p2,
                        body: b2,
                        ..
                    },
                ) => {
                    if p1 != p2 {
                        return false;
                    }

                    pending.push(Pending::Terms(b1, b2));
                }
                (
                    Term::Apply {
                        function: f1,
                        argument: a1,
                        ..
                    },
                    Term::Apply {
                        function: f2,
                        argument: a2,
                        ..
                    },
                ) => {
                    pending.push(Pending::Terms(a1, a2));
                    pending.push(Pending::Terms(f1, f2));
                }
                (Term::Constant { value: v1, .. }, Term::Constant { value: v2, .. }) => {
                    if v1 != v2 {
                        return false;
                    }
                }
                (Term::Error { .. }, Term::Error { .. }) => {}
                (Term::Builtin { fun: f1, .. }, Term::Builtin { fun: f2, .. }) => {
                    if f1 != f2 {
                        return false;
                    }
                }
                (
                    Term::Constr {
                        tag: t1,
                        fields: f1,
                        ..
                    },
                    Term::Constr {
                        tag: t2,
                        fields: f2,
                        ..
                    },
                ) => {
                    if t1 != t2 || f1.len() != f2.len() {
                        return false;
                    }

                    pending.extend(
                        f1.iter()
                            .zip(f2.iter())
                            .rev()
                            .map(|(x, y)| Pending::Terms(x, y)),
                    );
                }
                (
                    Term::Case {
                        constr: c1,
                        branches: b1,
                        ..
                    },
                    Term::Case {
                        constr: c2,
                        branches: b2,
                        ..
                    },
                ) => {
                    // The scrutinees are compared before the branches, whose lengths are only
                    // looked at then.
                    if b1.len() == b2.len() {
                        pending.extend(
                            b1.iter()
                                .zip(b2.iter())
                                .rev()
                                .map(|(x, y)| Pending::Terms(x, y)),
                        );
                    } else {
                        pending.push(Pending::Check(false));
                    }

                    pending.push(Pending::Terms(c1, c2));
                }
                _ => return false,
            }
        }

        true
    }
}

impl<T> Term<T> {
    /// Change a constant integer, possibly under `delay`s and `force`s, to its opposite.
    pub fn try_negate(&self) -> Option<Self> {
        // The wrappers on the way down, outermost first, rebuilt innermost first: the same order
        // a recursive rebuild hands out fresh ids in.
        let mut delays = Vec::new();
        let mut term = self;

        let mut negated = loop {
            match term {
                Self::Constant { value: cst, .. } => match cst.as_ref() {
                    Constant::Integer(i) => {
                        break Self::Constant {
                            value: Rc::new(Constant::Integer(-1 * i)),
                            uniq_id: next_uniq_id(),
                        };
                    }
                    _ => return None,
                },
                Self::Delay { body, .. } => {
                    delays.push(true);
                    term = body;
                }
                Self::Force { body, .. } => {
                    delays.push(false);
                    term = body;
                }
                _ => return None,
            }
        };

        while let Some(is_delay) = delays.pop() {
            let body = Rc::new(negated);

            negated = if is_delay {
                Self::Delay {
                    body,
                    uniq_id: next_uniq_id(),
                }
            } else {
                Self::Force {
                    body,
                    uniq_id: next_uniq_id(),
                }
            };
        }

        Some(negated)
    }
}

impl<T> TryInto<PlutusData> for Term<T> {
    type Error = String;

    fn try_into(self) -> Result<PlutusData, String> {
        // `Term` has a manual `Drop` (see below), so its fields cannot be moved out by pattern.
        // Nothing is moved out here anyway: the data is copied out of the borrowed constant.
        match &self {
            Term::Constant { value: rc, .. } => match rc.as_ref() {
                Constant::Data(data) => Ok(data.to_owned()),
                _ => Err("not a data".to_string()),
            },
            _ => Err("not a data".to_string()),
        }
    }
}

impl<'a, T> Display for Term<T>
where
    T: Binder<'a>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_pretty())
    }
}

/// The compiler-derived destructor for `Term` is recursive: dropping a node drops its `Rc`
/// children, which drop theirs, one stack frame per level of nesting. The nesting depth of a
/// term is attacker-controlled — the cheapest flat encoding of one level is a single 4-bit
/// `delay` (or `force`) tag, so a script that fits inside the on-chain size limit is already
/// tens of thousands of levels deep — and on `wasm32` the engine's call stack is about a
/// megabyte and cannot be grown from the page. `Term::decode` is iterative for exactly that
/// reason; without this impl, a term that decodes fine still blows the stack when it is
/// released. Do not remove this as redundant.
///
/// The teardown moves every child out of the node it is dismantling and onto an explicit
/// worklist, so the stack stays O(1) no matter how deep the tree is.
impl<T> Drop for Term<T> {
    fn drop(&mut self) {
        let mut pending: Vec<Term<T>> = Vec::new();
        let mut filler: Option<Rc<Term<T>>> = None;

        detach_term_children(self, &mut pending, &mut filler);

        while let Some(mut term) = pending.pop() {
            detach_term_children(&mut term, &mut pending, &mut filler);

            // Every child `term` owned has been moved onto `pending`, so the `Drop` that runs
            // here is shallow: it re-enters this impl once, finds nothing left to detach, and
            // returns without recursing any further.
        }
    }
}

/// Move the children `term` solely owns onto `pending`, leaving `term` childless.
fn detach_term_children<T>(
    term: &mut Term<T>,
    pending: &mut Vec<Term<T>>,
    filler: &mut Option<Rc<Term<T>>>,
) {
    match term {
        Term::Var { .. }
        | Term::Constant { .. }
        | Term::Error { .. }
        | Term::Builtin { .. } => {}

        Term::Delay { body, .. } | Term::Lambda { body, .. } | Term::Force { body, .. } => {
            detach_term_child(body, pending, filler);
        }

        Term::Apply {
            function, argument, ..
        } => {
            detach_term_child(function, pending, filler);
            detach_term_child(argument, pending, filler);
        }

        Term::Constr { fields, .. } => {
            // Fields are owned outright, so they always come along.
            pending.append(fields);
        }

        Term::Case {
            constr, branches, ..
        } => {
            detach_term_child(constr, pending, filler);
            pending.append(branches);
        }
    }
}

/// Take the subtree behind `slot`, but only when we are its sole owner.
///
/// `Rc` children may be shared, and a shared subtree is not ours to dismantle: unwrapping it
/// would strip it out from under the other holders. So we look at the strong count first — the
/// exact condition under which `Rc::try_unwrap` hands the value back — and when the handle is
/// shared we leave it where it is. Dropping a shared `Rc` along with its node is O(1) and never
/// touches the pointee, so nothing recurses.
///
/// When we *are* the sole owner we swap in a leaf before unwrapping, so the node we are
/// dismantling is left holding something harmless. That stand-in is a clone of a single `Error`
/// leaf shared by the whole teardown, which means the shallow `Drop` that later visits those
/// slots sees a strong count above one and skips them — no allocation, no extra work per node.
fn detach_term_child<T>(
    slot: &mut Rc<Term<T>>,
    pending: &mut Vec<Term<T>>,
    filler: &mut Option<Rc<Term<T>>>,
) {
    if Rc::strong_count(slot) != 1 {
        return;
    }

    // The leaf never escapes this teardown, so its id is never observed.
    let filler = filler.get_or_insert_with(|| Rc::new(Term::Error { uniq_id: 0 }));

    if let Ok(term) = Rc::try_unwrap(mem::replace(slot, Rc::clone(filler))) {
        pending.push(term);
    }
}

/// A container for the various constants that are available
/// in Untyped Plutus Core. Used in the `Constant` variant of `Term`.
#[derive(Clone)]
pub enum Constant {
    // tag: 0
    Integer(BigInt),
    // tag: 1
    ByteString(Vec<u8>),
    // tag: 2
    String(String),
    // tag: 3
    Unit,
    // tag: 4
    Bool(bool),
    // tag: 5
    // Elements are `Rc`-shared so list builtins (mkCons, tailList) can build
    // derived lists without deep-cloning every element; see `deep_clone` for
    // the thread-isolation caveat.
    ProtoList(Type, Vec<Rc<Constant>>),
    // tag: 6
    ProtoPair(Type, Type, Rc<Constant>, Rc<Constant>),
    // tag: 7
    // Apply(Box<Constant>, Type),
    // tag: 8
    Data(PlutusData),
    Bls12_381G1Element(Box<blst::blst_p1>),
    Bls12_381G2Element(Box<blst::blst_p2>),
    Bls12_381MlResult(Box<blst::blst_fp12>),
    // tag: 12
    // Elements are `Rc`-shared for the same reason as `ProtoList`'s: `listToArray` and
    // `indexArray` hand them over without deep-cloning them.
    ProtoArray(Type, Vec<Rc<Constant>>),
    // tag: 13
    Value(Value),
}

impl Constant {
    /// Whether this constant contains a native `Value`, which is only
    /// available from protocol version 11 (Van Rossem) onwards.
    pub fn contains_value(constant: impl AsRef<Constant>) -> bool {
        constant.as_ref().mentions(
            |constant| matches!(constant, Constant::Value(_)),
            |typ| matches!(typ, Type::Value),
        )
    }

    /// Whether this constant contains an array, which is only available from
    /// protocol version 11 (Van Rossem) onwards.
    pub fn contains_array(constant: impl AsRef<Constant>) -> bool {
        constant.as_ref().mentions(
            |constant| matches!(constant, Constant::ProtoArray(..)),
            |typ| matches!(typ, Type::Array(_)),
        )
    }

    /// Whether this constant, one nested in it, or any type they carry (declared types included,
    /// so a type-inconsistent constant built by hand cannot hide anything) matches.
    ///
    /// Iterative, for the depth reason given on the `Drop` impl below. Shared nodes are visited
    /// once: a nested constant's type shares its `Rc` children with the enclosing type, so
    /// walking each declared type in full would cost the square of the depth, and constants built
    /// as DAGs could cost exponentially more.
    fn mentions(
        &self,
        constant_matches: impl Fn(&Constant) -> bool,
        type_matches: impl Fn(&Type) -> bool,
    ) -> bool {
        let mut constants = vec![self];
        let mut types: Vec<&Type> = Vec::new();
        let mut seen_constants: HashSet<*const Constant> = HashSet::new();
        let mut seen_types: HashSet<*const Type> = HashSet::new();

        while let Some(constant) = constants.pop() {
            if constant_matches(constant) {
                return true;
            }

            match constant {
                Constant::ProtoList(r#type, elements) | Constant::ProtoArray(r#type, elements) => {
                    types.push(r#type);

                    for element in elements {
                        if seen_constants.insert(Rc::as_ptr(element)) {
                            constants.push(element);
                        }
                    }
                }
                Constant::ProtoPair(fst, snd, left, right) => {
                    types.push(fst);
                    types.push(snd);

                    for element in [left, right] {
                        if seen_constants.insert(Rc::as_ptr(element)) {
                            constants.push(element);
                        }
                    }
                }
                _ => {}
            }

            if Type::any_in(&mut types, &mut seen_types, &type_matches) {
                return true;
            }
        }

        false
    }
}

pub const VALUE_MAX_KEY_LEN: usize = 32;
pub const VALUE_DATA_MAX_SIZE: usize = 40_000;

pub type ValueEntry<Quantity> = (Vec<u8>, Vec<(Vec<u8>, Quantity)>);
pub type ValueEntries = Vec<ValueEntry<i128>>;
pub type BigIntValueEntries = Vec<ValueEntry<BigInt>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    KeyTooLong(usize),
    QuantityOutOfBounds(BigInt),
    DataQuantityOutOfBounds,
    CurrencySymbolsNotStrictlyAscending,
    TokenNamesNotStrictlyAscending,
    EmptyInnerMap,
    ZeroQuantity,
    FirstValueContainsNegativeAmounts,
    SecondValueContainsNegativeAmounts,
    ExpectedDataMap,
    ExpectedDataBytes,
    ExpectedDataInteger,
    ValueDataInputTooLarge(usize),
}

impl Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyTooLong(length) => write!(
                f,
                "Value key exceeds maximum length of {VALUE_MAX_KEY_LEN} bytes: got {length} bytes"
            ),
            Self::QuantityOutOfBounds(quantity) => write!(
                f,
                "Value quantity out of signed 128-bit integer bounds: {quantity}"
            ),
            Self::DataQuantityOutOfBounds => {
                f.write_str("Value quantity out of signed 128-bit integer bounds")
            }
            Self::CurrencySymbolsNotStrictlyAscending => {
                f.write_str("Value currency symbols are not strictly ascending")
            }
            Self::TokenNamesNotStrictlyAscending => {
                f.write_str("Value token names are not strictly ascending")
            }
            Self::EmptyInnerMap => f.write_str("Value contains an empty inner map"),
            Self::ZeroQuantity => f.write_str("Value contains a zero quantity"),
            Self::FirstValueContainsNegativeAmounts => {
                f.write_str("valueContains: first value contains negative amounts")
            }
            Self::SecondValueContainsNegativeAmounts => {
                f.write_str("valueContains: second value contains negative amounts")
            }
            Self::ExpectedDataMap => f.write_str("unValueData: non-Map constructor"),
            Self::ExpectedDataBytes => f.write_str("unValueData: non-B constructor"),
            Self::ExpectedDataInteger => f.write_str("unValueData: non-I constructor"),
            Self::ValueDataInputTooLarge(size) => write!(
                f,
                "valueData: maximum input size ({VALUE_DATA_MAX_SIZE}) exceeded: got {size}"
            ),
        }
    }
}

impl std::error::Error for ValueError {}

type ValueInner = BTreeMap<Vec<u8>, i128>;
type ValueMap = BTreeMap<Vec<u8>, ValueInner>;

#[cfg(test)]
std::thread_local! {
    static VALUE_DATA_KEY_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, Clone)]
pub struct Value {
    entries: ValueMap,
    total_size: usize,
    max_inner_size: usize,
    negative_amounts: usize,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Eq for Value {}

impl Default for Value {
    fn default() -> Self {
        Self::empty()
    }
}

impl Value {
    pub fn empty() -> Self {
        Self {
            entries: ValueMap::new(),
            total_size: 0,
            max_inner_size: 0,
            negative_amounts: 0,
        }
    }

    pub fn from_canonical_entries(entries: BigIntValueEntries) -> Result<Self, ValueError> {
        Self::from_strict_entries(entries, BigInt::is_zero, |quantity| {
            i128::try_from(&quantity).map_err(|_| ValueError::QuantityOutOfBounds(quantity))
        })
    }

    pub fn from_canonical_bounded_entries(entries: ValueEntries) -> Result<Self, ValueError> {
        Self::from_strict_entries(entries, |quantity| *quantity == 0, Ok)
    }

    fn from_strict_entries<Quantity>(
        entries: Vec<ValueEntry<Quantity>>,
        is_zero: impl Fn(&Quantity) -> bool,
        into_bounded: impl Fn(Quantity) -> Result<i128, ValueError>,
    ) -> Result<Self, ValueError> {
        let mut canonical = ValueEntries::with_capacity(entries.len());

        for (currency, tokens) in entries {
            Self::check_key(&currency)?;
            if canonical
                .last()
                .is_some_and(|(previous, _)| previous.as_slice() >= currency.as_slice())
            {
                return Err(ValueError::CurrencySymbolsNotStrictlyAscending);
            }
            if tokens.is_empty() {
                return Err(ValueError::EmptyInnerMap);
            }

            let mut inner: Vec<(Vec<u8>, i128)> = Vec::with_capacity(tokens.len());

            for (token, quantity) in tokens {
                Self::check_key(&token)?;
                if inner
                    .last()
                    .is_some_and(|(previous, _)| previous.as_slice() >= token.as_slice())
                {
                    return Err(ValueError::TokenNamesNotStrictlyAscending);
                }
                if is_zero(&quantity) {
                    return Err(ValueError::ZeroQuantity);
                }

                inner.push((token, into_bounded(quantity)?));
            }

            canonical.push((currency, inner));
        }

        Ok(Self::from_normalized(canonical))
    }

    pub(crate) fn iter(
        &self,
    ) -> impl Iterator<Item = (&Vec<u8>, impl Iterator<Item = (&Vec<u8>, &i128)>)> {
        self.entries
            .iter()
            .map(|(currency, inner)| (currency, inner.iter()))
    }

    pub fn into_entries(self) -> ValueEntries {
        self.entries
            .into_iter()
            .map(|(currency, inner)| (currency, inner.into_iter().collect()))
            .collect()
    }

    pub fn total_size(&self) -> usize {
        self.total_size
    }

    pub fn max_inner_size(&self) -> usize {
        self.max_inner_size
    }

    pub fn outer_size(&self) -> usize {
        self.entries.len()
    }

    pub fn negative_amounts(&self) -> usize {
        self.negative_amounts
    }

    pub fn insert_coin(
        &self,
        currency: &[u8],
        token: &[u8],
        quantity: &BigInt,
    ) -> Result<Self, ValueError> {
        if quantity.is_zero() {
            if self
                .entries
                .get(currency)
                .is_none_or(|inner| !inner.contains_key(token))
            {
                return Ok(self.clone());
            }

            let mut entries = self.entries.clone();
            let inner = entries
                .get_mut(currency)
                .expect("deleted coin's currency exists in the value");
            inner.remove(token);
            if inner.is_empty() {
                entries.remove(currency);
            }

            return Ok(Self::from_map(entries));
        }

        Self::check_key(currency)?;
        Self::check_key(token)?;
        let quantity = i128::try_from(quantity)
            .map_err(|_| ValueError::QuantityOutOfBounds(quantity.clone()))?;

        if self
            .entries
            .get(currency)
            .and_then(|inner| inner.get(token))
            == Some(&quantity)
        {
            return Ok(self.clone());
        }

        let mut entries = self.entries.clone();
        entries
            .entry(currency.to_vec())
            .or_default()
            .insert(token.to_vec(), quantity);

        Ok(Self::from_map(entries))
    }

    pub fn lookup_coin(&self, currency: &[u8], token: &[u8]) -> i128 {
        self.entries
            .get(currency)
            .and_then(|inner| inner.get(token))
            .copied()
            .unwrap_or(0)
    }

    pub fn union(&self, other: &Self) -> Result<Self, ValueError> {
        if self.total_size == 0 {
            return Ok(other.clone());
        }
        if other.total_size == 0 {
            return Ok(self.clone());
        }

        let mut entries = self.entries.clone();

        for (currency, right_tokens) in &other.entries {
            let merged = match entries.get(currency) {
                Some(left_tokens) => Self::union_inner(left_tokens, right_tokens)?,
                None => right_tokens.clone(),
            };
            if merged.is_empty() {
                entries.remove(currency);
            } else {
                entries.insert(currency.clone(), merged);
            }
        }

        Ok(Self::from_map(entries))
    }

    fn union_inner(left: &ValueInner, right: &ValueInner) -> Result<ValueInner, ValueError> {
        let mut merged = left.clone();

        for (token, quantity) in right {
            let existing = merged.get(token).copied().unwrap_or(0);
            let combined = existing.checked_add(*quantity).ok_or_else(|| {
                ValueError::QuantityOutOfBounds(BigInt::from(existing) + BigInt::from(*quantity))
            })?;
            if combined == 0 {
                merged.remove(token);
            } else {
                merged.insert(token.clone(), combined);
            }
        }

        Ok(merged)
    }

    pub fn contains(&self, other: &Self) -> Result<bool, ValueError> {
        if self.negative_amounts != 0 {
            return Err(ValueError::FirstValueContainsNegativeAmounts);
        }
        if other.negative_amounts != 0 {
            return Err(ValueError::SecondValueContainsNegativeAmounts);
        }
        if self.total_size < other.total_size {
            return Ok(false);
        }

        Ok(other.entries.iter().all(|(currency, tokens)| {
            tokens
                .iter()
                .all(|(token, quantity)| self.lookup_coin(currency, token) >= *quantity)
        }))
    }

    pub fn scale(&self, scalar: &BigInt) -> Result<Self, ValueError> {
        if scalar.is_zero() {
            return Ok(Self::empty());
        }
        if scalar.is_one() {
            return Ok(self.clone());
        }

        let mut entries = Vec::with_capacity(self.entries.len());
        for (currency, tokens) in self.entries.iter() {
            let mut inner = Vec::with_capacity(tokens.len());
            for (token, quantity) in tokens.iter() {
                let product = scalar * BigInt::from(*quantity);
                let bounded = i128::try_from(&product)
                    .map_err(|_| ValueError::QuantityOutOfBounds(product))?;
                inner.push((token.clone(), bounded));
            }
            entries.push((currency.clone(), inner));
        }

        Ok(Self::from_normalized(entries))
    }

    fn to_data_unchecked(&self) -> PlutusData {
        Data::map(
            self.entries
                .iter()
                .map(|(currency, tokens)| {
                    (
                        Data::bytestring(currency.clone()),
                        Data::map(
                            tokens
                                .iter()
                                .map(|(token, quantity)| {
                                    (
                                        Data::bytestring(token.clone()),
                                        Data::integer(BigInt::from(*quantity)),
                                    )
                                })
                                .collect(),
                        ),
                    )
                })
                .collect(),
        )
    }

    pub fn to_data_checked(&self) -> Result<PlutusData, ValueError> {
        if self.total_size > VALUE_DATA_MAX_SIZE {
            Err(ValueError::ValueDataInputTooLarge(self.total_size))
        } else {
            Ok(self.to_data_unchecked())
        }
    }

    pub fn from_data(data: &PlutusData) -> Result<Self, ValueError> {
        let PlutusData::Map(outer) = data else {
            return Err(ValueError::ExpectedDataMap);
        };
        let mut entries = ValueEntries::with_capacity(outer.len());

        for (currency, tokens) in outer.iter() {
            let PlutusData::BoundedBytes(currency) = currency else {
                return Err(ValueError::ExpectedDataBytes);
            };
            Self::check_key(currency)?;

            let PlutusData::Map(tokens) = tokens else {
                return Err(ValueError::ExpectedDataMap);
            };

            if entries
                .last()
                .is_some_and(|(previous, _)| previous.as_slice() >= currency.as_slice())
            {
                return Err(ValueError::CurrencySymbolsNotStrictlyAscending);
            }

            let mut inner: Vec<(Vec<u8>, i128)> = Vec::with_capacity(tokens.len());
            for (token, quantity) in tokens.iter() {
                let PlutusData::BoundedBytes(token) = token else {
                    return Err(ValueError::ExpectedDataBytes);
                };
                Self::check_key(token)?;

                let PlutusData::BigInt(quantity) = quantity else {
                    return Err(ValueError::ExpectedDataInteger);
                };
                let quantity = pallas_bigint_to_i128(quantity)?;

                if inner
                    .last()
                    .is_some_and(|(previous, _)| previous.as_slice() >= token.as_slice())
                {
                    return Err(ValueError::TokenNamesNotStrictlyAscending);
                }
                if quantity == 0 {
                    return Err(ValueError::ZeroQuantity);
                }

                inner.push((Self::clone_data_key(token), quantity));
            }

            if inner.is_empty() {
                return Err(ValueError::EmptyInnerMap);
            }

            entries.push((Self::clone_data_key(currency), inner));
        }

        Ok(Self::from_normalized(entries))
    }

    fn check_key(key: &[u8]) -> Result<(), ValueError> {
        if key.len() > VALUE_MAX_KEY_LEN {
            Err(ValueError::KeyTooLong(key.len()))
        } else {
            Ok(())
        }
    }

    fn clone_data_key(key: &[u8]) -> Vec<u8> {
        #[cfg(test)]
        VALUE_DATA_KEY_COPIES.with(|copies| copies.set(copies.get() + 1));

        key.to_vec()
    }

    fn from_normalized(entries: ValueEntries) -> Self {
        Self::from_map(
            entries
                .into_iter()
                .map(|(currency, inner)| (currency, inner.into_iter().collect()))
                .collect(),
        )
    }

    fn from_map(entries: ValueMap) -> Self {
        let mut total_size = 0;
        let mut max_inner_size = 0;
        let mut negative_amounts = 0;

        for inner in entries.values() {
            total_size += inner.len();
            max_inner_size = max_inner_size.max(inner.len());
            negative_amounts += inner
                .values()
                .filter(|quantity| quantity.is_negative())
                .count();
        }

        Self {
            entries,
            total_size,
            max_inner_size,
            negative_amounts,
        }
    }

    #[cfg(test)]
    fn reset_data_key_copy_count() {
        VALUE_DATA_KEY_COPIES.with(|copies| copies.set(0));
    }

    #[cfg(test)]
    fn data_key_copy_count() -> usize {
        VALUE_DATA_KEY_COPIES.with(std::cell::Cell::get)
    }
}

fn pallas_bigint_to_i128(quantity: &conway::BigInt) -> Result<i128, ValueError> {
    let magnitude = match quantity {
        conway::BigInt::Int(quantity) => return Ok(i128::from(*quantity)),
        conway::BigInt::BigUInt(bytes) | conway::BigInt::BigNInt(bytes) => bytes,
    };
    let first_nonzero = magnitude
        .iter()
        .position(|byte| *byte != 0)
        .unwrap_or(magnitude.len());
    let magnitude = &magnitude[first_nonzero..];

    if magnitude.len() > i128::BITS as usize / 8
        || (magnitude.len() == i128::BITS as usize / 8 && magnitude[0] & 0x80 != 0)
    {
        return Err(ValueError::DataQuantityOutOfBounds);
    }

    i128::try_from(crate::machine::value::from_pallas_bigint(quantity))
        .map_err(|_| ValueError::DataQuantityOutOfBounds)
}

/// `Constant` nests through `ProtoList` and `ProtoPair`, and its depth is attacker-controlled
/// the same way `Term`'s is: the element type is a run of 4-bit tags, so `list (list (list …))`
/// costs a couple of bytes per level and an in-limit script reaches thousands of levels. See the
/// `Drop` impl on `Term` for the full rationale; the teardown here works the same way.
impl Drop for Constant {
    fn drop(&mut self) {
        let mut pending: Vec<Constant> = Vec::new();
        let mut filler: Option<Rc<Constant>> = None;

        detach_constant_children(self, &mut pending, &mut filler);

        while let Some(mut constant) = pending.pop() {
            detach_constant_children(&mut constant, &mut pending, &mut filler);
        }
    }
}

/// Move the constants `constant` solely owns onto `pending`, leaving it childless.
///
/// The `Type` fields are left in place on purpose: they carry their own iterative `Drop`, so the
/// shallow drop of this node releases them without recursing.
fn detach_constant_children(
    constant: &mut Constant,
    pending: &mut Vec<Constant>,
    filler: &mut Option<Rc<Constant>>,
) {
    match constant {
        Constant::Integer(..)
        | Constant::ByteString(..)
        | Constant::String(..)
        | Constant::Unit
        | Constant::Bool(..)
        | Constant::Data(..)
        | Constant::Bls12_381G1Element(..)
        | Constant::Bls12_381G2Element(..)
        | Constant::Bls12_381MlResult(..)
        | Constant::Value(..) => {}

        Constant::ProtoList(_, items) | Constant::ProtoArray(_, items) => {
            // Elements are shared `Rc`s: only the ones we solely own are taken apart, the rest are
            // released in O(1) without touching the pointee.
            for item in mem::take(items) {
                if let Ok(item) = Rc::try_unwrap(item) {
                    pending.push(item);
                }
            }
        }

        Constant::ProtoPair(_, _, fst, snd) => {
            detach_constant_child(fst, pending, filler);
            detach_constant_child(snd, pending, filler);
        }
    }
}

/// Take the constant behind `slot`, but only when we are its sole owner — same `Rc` reasoning as
/// [`detach_term_child`].
fn detach_constant_child(
    slot: &mut Rc<Constant>,
    pending: &mut Vec<Constant>,
    filler: &mut Option<Rc<Constant>>,
) {
    if Rc::strong_count(slot) != 1 {
        return;
    }

    let filler = filler.get_or_insert_with(|| Rc::new(Constant::Unit));

    if let Ok(constant) = Rc::try_unwrap(mem::replace(slot, Rc::clone(filler))) {
        pending.push(constant);
    }
}

pub struct Data;

// TODO: See about moving these builders upstream to Pallas?
impl Data {
    pub fn to_hex(data: PlutusData) -> String {
        let mut bytes = Vec::new();
        pallas_codec::minicbor::Encoder::new(&mut bytes)
            .encode(data)
            .expect("failed to encode Plutus Data as cbor?");
        hex::encode(bytes)
    }

    pub fn integer(i: BigInt) -> PlutusData {
        PlutusData::BigInt(to_pallas_bigint(&i))
    }

    pub fn bytestring(bytes: Vec<u8>) -> PlutusData {
        PlutusData::BoundedBytes(bytes.into())
    }

    pub fn map(kvs: Vec<(PlutusData, PlutusData)>) -> PlutusData {
        PlutusData::Map(kvs.into())
    }

    pub fn list(xs: Vec<PlutusData>) -> PlutusData {
        PlutusData::Array(if xs.is_empty() {
            conway::MaybeIndefArray::Def(xs)
        } else {
            conway::MaybeIndefArray::Indef(xs)
        })
    }

    pub fn constr(ix: u64, fields: Vec<PlutusData>) -> PlutusData {
        let fields = if fields.is_empty() {
            conway::MaybeIndefArray::Def(fields)
        } else {
            conway::MaybeIndefArray::Indef(fields)
        };

        // NOTE: see https://github.com/input-output-hk/plutus/blob/9538fc9829426b2ecb0628d352e2d7af96ec8204/plutus-core/plutus-core/src/PlutusCore/Data.hs#L139-L155
        if ix < 7 {
            PlutusData::Constr(Constr {
                tag: 121 + ix,
                any_constructor: None,
                fields,
            })
        } else if ix < 128 {
            PlutusData::Constr(Constr {
                tag: 1280 + ix - 7,
                any_constructor: None,
                fields,
            })
        } else {
            PlutusData::Constr(Constr {
                tag: 102,
                any_constructor: Some(ix),
                fields,
            })
        }
    }
}

pub enum Type {
    Bool,
    Integer,
    String,
    ByteString,
    Unit,
    List(Rc<Type>),
    Pair(Rc<Type>, Rc<Type>),
    Data,
    Bls12_381G1Element,
    Bls12_381G2Element,
    Bls12_381MlResult,
    Value,
    Array(Rc<Type>),
}

impl Type {
    /// Whether this type mentions the native `Value` type, which is only
    /// available from protocol version 11 (Van Rossem) onwards.
    pub fn contains_value(&self) -> bool {
        Type::any_in(&mut vec![self], &mut HashSet::new(), &|typ| {
            matches!(typ, Type::Value)
        })
    }

    /// Whether this type mentions the `array` type, which is only available
    /// from protocol version 11 (Van Rossem) onwards.
    pub fn contains_array(&self) -> bool {
        Type::any_in(&mut vec![self], &mut HashSet::new(), &|typ| {
            matches!(typ, Type::Array(_))
        })
    }

    /// Whether a type on `pending`, or one nested in it, matches; drains `pending`. Iterative,
    /// for the depth reason given on the `Drop` impl below, and visiting each shared `Rc` child
    /// once (it is recorded in `seen`), so a type shaped as a DAG costs no more than its size.
    fn any_in<'a>(
        pending: &mut Vec<&'a Type>,
        seen: &mut HashSet<*const Type>,
        matches: &impl Fn(&Type) -> bool,
    ) -> bool {
        while let Some(typ) = pending.pop() {
            if matches(typ) {
                pending.clear();
                return true;
            }

            match typ {
                Type::List(inner) | Type::Array(inner) => {
                    if seen.insert(Rc::as_ptr(inner)) {
                        pending.push(inner);
                    }
                }
                Type::Pair(fst, snd) => {
                    for inner in [fst, snd] {
                        if seen.insert(Rc::as_ptr(inner)) {
                            pending.push(inner);
                        }
                    }
                }
                _ => {}
            }
        }

        false
    }
}

impl Constant {
    /// An equal constant sharing no allocation with `self`.
    ///
    /// `Rc` reference counts are not atomic, so a constant that may end up
    /// embedded in programs handled by different threads (e.g. tests run in
    /// parallel) must not share `Rc`s with anything else.
    ///
    /// Iterative, for the depth reason given on the `Drop` impl below: `copies` holds the finished
    /// copies of the elements whose enclosing constant is still waiting on `tasks`.
    pub fn deep_clone(&self) -> Constant {
        enum Task<'a> {
            Copy(&'a Constant),
            /// Rebuild this list, array or pair around the last copies made, its elements.
            Assemble(&'a Constant),
        }

        let mut tasks = vec![Task::Copy(self)];
        let mut copies: Vec<Rc<Constant>> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Copy(constant) => match constant {
                    Constant::ProtoList(_, items) | Constant::ProtoArray(_, items) => {
                        tasks.push(Task::Assemble(constant));
                        tasks.extend(items.iter().rev().map(|item| Task::Copy(item)));
                    }
                    Constant::ProtoPair(_, _, fst, snd) => {
                        tasks.push(Task::Assemble(constant));
                        tasks.push(Task::Copy(snd));
                        tasks.push(Task::Copy(fst));
                    }
                    // The remaining variants own their contents outright.
                    other => copies.push(Rc::new(other.clone())),
                },
                Task::Assemble(constant) => {
                    let copy = match constant {
                        Constant::ProtoList(tipo, items) => Constant::ProtoList(
                            tipo.deep_clone(),
                            copies.split_off(copies.len() - items.len()),
                        ),
                        Constant::ProtoArray(tipo, items) => Constant::ProtoArray(
                            tipo.deep_clone(),
                            copies.split_off(copies.len() - items.len()),
                        ),
                        Constant::ProtoPair(fst_tipo, snd_tipo, _, _) => {
                            let snd = copies.pop().expect("second component copied");
                            let fst = copies.pop().expect("first component copied");

                            Constant::ProtoPair(
                                fst_tipo.deep_clone(),
                                snd_tipo.deep_clone(),
                                fst,
                                snd,
                            )
                        }
                        _ => unreachable!("only lists, arrays and pairs are assembled"),
                    };

                    copies.push(Rc::new(copy));
                }
            }
        }

        let root = copies.pop().expect("the copy of the root");

        Rc::try_unwrap(root).unwrap_or_else(|_| unreachable!("the root copy is not shared"))
    }
}

impl Type {
    /// An equal type sharing no allocation with `self` (see
    /// `Constant::deep_clone`).
    ///
    /// Iterative, for the depth reason given on the `Drop` impl below.
    pub fn deep_clone(&self) -> Type {
        enum Task<'a> {
            Copy(&'a Type),
            /// Rebuild this list, array or pair around the last copies made, its operands.
            Assemble(&'a Type),
        }

        let mut tasks = vec![Task::Copy(self)];
        let mut copies: Vec<Type> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Copy(typ) => match typ {
                    Type::List(inner) | Type::Array(inner) => {
                        tasks.push(Task::Assemble(typ));
                        tasks.push(Task::Copy(inner));
                    }
                    Type::Pair(fst, snd) => {
                        tasks.push(Task::Assemble(typ));
                        tasks.push(Task::Copy(snd));
                        tasks.push(Task::Copy(fst));
                    }
                    other => copies.push(other.clone()),
                },
                Task::Assemble(typ) => {
                    let copy = match typ {
                        Type::List(_) => Type::List(Rc::new(copies.pop().expect("operand copied"))),
                        Type::Array(_) => {
                            Type::Array(Rc::new(copies.pop().expect("operand copied")))
                        }
                        Type::Pair(_, _) => {
                            let snd = Rc::new(copies.pop().expect("second operand copied"));
                            let fst = Rc::new(copies.pop().expect("first operand copied"));

                            Type::Pair(fst, snd)
                        }
                        _ => unreachable!("only lists, arrays and pairs are assembled"),
                    };

                    copies.push(copy);
                }
            }
        }

        copies.pop().expect("the copy of the root")
    }
}

/// The derived `Clone` copies a whole type tree, one stack frame per level, and the flat decoder
/// clones the element type at every level of a nested `list` -- so a deep constant would overflow
/// on the way in. Cloning the top node and sharing its `Rc` children instead is O(1) and
/// unobservable: a `Type` is immutable once built (nothing in the crate reaches through an
/// `Rc<Type>` with `make_mut` or `get_mut`, and there is no interior mutability), and `PartialEq`
/// and `Debug` on `Rc` both read straight through to the pointee. Do not put this back on the
/// derive.
impl Clone for Type {
    fn clone(&self) -> Self {
        match self {
            Type::Bool => Type::Bool,
            Type::Integer => Type::Integer,
            Type::String => Type::String,
            Type::ByteString => Type::ByteString,
            Type::Unit => Type::Unit,
            Type::Data => Type::Data,
            Type::Bls12_381G1Element => Type::Bls12_381G1Element,
            Type::Bls12_381G2Element => Type::Bls12_381G2Element,
            Type::Bls12_381MlResult => Type::Bls12_381MlResult,
            Type::Value => Type::Value,
            Type::List(inner) => Type::List(Rc::clone(inner)),
            Type::Array(inner) => Type::Array(Rc::clone(inner)),
            Type::Pair(fst, snd) => Type::Pair(Rc::clone(fst), Rc::clone(snd)),
        }
    }
}

/// `Type` nests through `List` and `Pair`, and a deep constant carries an equally deep element
/// type along with it, so the derived destructor would recurse just as far. See the `Drop` impl
/// on `Term` for the full rationale.
impl Drop for Type {
    fn drop(&mut self) {
        let mut pending: Vec<Type> = Vec::new();
        let mut filler: Option<Rc<Type>> = None;

        detach_type_children(self, &mut pending, &mut filler);

        while let Some(mut typ) = pending.pop() {
            detach_type_children(&mut typ, &mut pending, &mut filler);
        }
    }
}

/// Move the types `typ` solely owns onto `pending`, leaving it childless.
fn detach_type_children(typ: &mut Type, pending: &mut Vec<Type>, filler: &mut Option<Rc<Type>>) {
    match typ {
        Type::Bool
        | Type::Integer
        | Type::String
        | Type::ByteString
        | Type::Unit
        | Type::Data
        | Type::Bls12_381G1Element
        | Type::Bls12_381G2Element
        | Type::Bls12_381MlResult
        | Type::Value => {}

        Type::List(inner) | Type::Array(inner) => {
            detach_type_child(inner, pending, filler);
        }

        Type::Pair(fst, snd) => {
            detach_type_child(fst, pending, filler);
            detach_type_child(snd, pending, filler);
        }
    }
}

/// Take the type behind `slot`, but only when we are its sole owner — same `Rc` reasoning as
/// [`detach_term_child`].
fn detach_type_child(slot: &mut Rc<Type>, pending: &mut Vec<Type>, filler: &mut Option<Rc<Type>>) {
    if Rc::strong_count(slot) != 1 {
        return;
    }

    let filler = filler.get_or_insert_with(|| Rc::new(Type::Unit));

    if let Ok(typ) = Rc::try_unwrap(mem::replace(slot, Rc::clone(filler))) {
        pending.push(typ);
    }
}

/// Iterative, for the depth reason given on the `Drop` impl above: `pending` holds the types and
/// separators still to write, the next one on top.
impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        enum Piece<'a> {
            Type(&'a Type),
            Space,
        }

        let mut pending = vec![Piece::Type(self)];

        while let Some(piece) = pending.pop() {
            let typ = match piece {
                Piece::Space => {
                    f.write_str(" ")?;
                    continue;
                }
                Piece::Type(typ) => typ,
            };

            match typ {
                Type::Bool => write!(f, "bool")?,
                Type::Integer => write!(f, "integer")?,
                Type::String => write!(f, "string")?,
                Type::ByteString => write!(f, "bytestring")?,
                Type::Unit => write!(f, "unit")?,
                Type::List(t) => {
                    write!(f, "list ")?;
                    pending.push(Piece::Type(t));
                }
                Type::Pair(t1, t2) => {
                    write!(f, "pair ")?;
                    pending.push(Piece::Type(t2));
                    pending.push(Piece::Space);
                    pending.push(Piece::Type(t1));
                }
                Type::Data => write!(f, "data")?,
                Type::Bls12_381G1Element => write!(f, "bls12_381_G1_element")?,
                Type::Bls12_381G2Element => write!(f, "bls12_381_G2_element")?,
                Type::Bls12_381MlResult => write!(f, "bls12_381_mlresult")?,
                Type::Value => write!(f, "value")?,
                Type::Array(t) => {
                    write!(f, "array ")?;
                    pending.push(Piece::Type(t));
                }
            }
        }

        Ok(())
    }
}

/// The derived format, `{:?}` and `{:#?}` alike, written without recursion (see `debug.rs`).
impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug::write(self, f, |typ| match typ {
            Type::Bool => Item::Tuple("Bool", vec![]),
            Type::Integer => Item::Tuple("Integer", vec![]),
            Type::String => Item::Tuple("String", vec![]),
            Type::ByteString => Item::Tuple("ByteString", vec![]),
            Type::Unit => Item::Tuple("Unit", vec![]),
            Type::List(inner) => Item::Tuple("List", vec![Item::Node(inner)]),
            Type::Pair(fst, snd) => Item::Tuple("Pair", vec![Item::Node(fst), Item::Node(snd)]),
            Type::Data => Item::Tuple("Data", vec![]),
            Type::Bls12_381G1Element => Item::Tuple("Bls12_381G1Element", vec![]),
            Type::Bls12_381G2Element => Item::Tuple("Bls12_381G2Element", vec![]),
            Type::Bls12_381MlResult => Item::Tuple("Bls12_381MlResult", vec![]),
            Type::Value => Item::Tuple("Value", vec![]),
            Type::Array(inner) => Item::Tuple("Array", vec![Item::Node(inner)]),
        })
    }
}

/// The derived format, `{:?}` and `{:#?}` alike, written without recursion (see `debug.rs`).
impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn elements(items: &[Rc<Constant>]) -> Item<'_, Constant> {
            Item::List(items.iter().map(|item| Item::Node(item.as_ref())).collect())
        }

        debug::write(self, f, |constant| match constant {
            Constant::Integer(i) => Item::Tuple("Integer", vec![Item::Leaf(i)]),
            Constant::ByteString(bytes) => Item::Tuple("ByteString", vec![Item::Leaf(bytes)]),
            Constant::String(s) => Item::Tuple("String", vec![Item::Leaf(s)]),
            Constant::Unit => Item::Tuple("Unit", vec![]),
            Constant::Bool(b) => Item::Tuple("Bool", vec![Item::Leaf(b)]),
            Constant::ProtoList(r#type, items) => {
                Item::Tuple("ProtoList", vec![Item::Leaf(r#type), elements(items)])
            }
            Constant::ProtoPair(fst_type, snd_type, fst, snd) => Item::Tuple(
                "ProtoPair",
                vec![
                    Item::Leaf(fst_type),
                    Item::Leaf(snd_type),
                    Item::Node(fst),
                    Item::Node(snd),
                ],
            ),
            Constant::Data(data) => Item::Tuple("Data", vec![Item::Leaf(data)]),
            Constant::Bls12_381G1Element(element) => {
                Item::Tuple("Bls12_381G1Element", vec![Item::Leaf(element)])
            }
            Constant::Bls12_381G2Element(element) => {
                Item::Tuple("Bls12_381G2Element", vec![Item::Leaf(element)])
            }
            Constant::Bls12_381MlResult(element) => {
                Item::Tuple("Bls12_381MlResult", vec![Item::Leaf(element)])
            }
            Constant::ProtoArray(r#type, items) => {
                Item::Tuple("ProtoArray", vec![Item::Leaf(r#type), elements(items)])
            }
            Constant::Value(value) => Item::Tuple("Value", vec![Item::Leaf(value)]),
        })
    }
}

/// Structural equality, as derived.
impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        types_equal(self, other, &mut HashSet::new())
    }
}

/// Structural equality of two types, iterative for the depth reason given on the `Drop` impl
/// above: `pending` holds the pairs of operands still to compare.
///
/// `seen` records the pairs of shared operands already compared (or queued), across calls: the
/// element types of a nested constant share their operands with the enclosing type, so without it
/// comparing two constants would compare the same deep operands once per level. Equality of types
/// is pure, so skipping a repeated pair cannot change the answer.
fn types_equal<'a>(
    left: &'a Type,
    right: &'a Type,
    seen: &mut HashSet<(*const Type, *const Type)>,
) -> bool {
    fn queue<'a>(
        pending: &mut Vec<(&'a Type, &'a Type)>,
        seen: &mut HashSet<(*const Type, *const Type)>,
        t1: &'a Rc<Type>,
        t2: &'a Rc<Type>,
    ) {
        if !Rc::ptr_eq(t1, t2) && seen.insert((Rc::as_ptr(t1), Rc::as_ptr(t2))) {
            pending.push((t1.as_ref(), t2.as_ref()));
        }
    }

    let mut pending = vec![(left, right)];

    while let Some(pair) = pending.pop() {
        match pair {
            (Type::List(t1), Type::List(t2)) | (Type::Array(t1), Type::Array(t2)) => {
                queue(&mut pending, seen, t1, t2);
            }
            (Type::Pair(l1, r1), Type::Pair(l2, r2)) => {
                queue(&mut pending, seen, r1, r2);
                queue(&mut pending, seen, l1, l2);
            }
            (Type::List(_) | Type::Array(_) | Type::Pair(_, _), _)
            | (_, Type::List(_) | Type::Array(_) | Type::Pair(_, _)) => return false,
            // The remaining variants carry nothing.
            (t1, t2) => {
                if mem::discriminant(t1) != mem::discriminant(t2) {
                    return false;
                }
            }
        }
    }

    true
}

/// Structural equality, as derived, but iterative for the depth reason given on the `Drop` impl
/// above: `pending` holds the pairs of elements still to compare. The types are compared with one
/// shared `seen` set, so a nested constant's types cost their size once, not once per level.
impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        let mut pending = vec![(self, other)];
        let mut seen_types = HashSet::new();

        while let Some(pair) = pending.pop() {
            match pair {
                (Constant::ProtoList(t1, xs), Constant::ProtoList(t2, ys))
                | (Constant::ProtoArray(t1, xs), Constant::ProtoArray(t2, ys)) => {
                    if !types_equal(t1, t2, &mut seen_types) || xs.len() != ys.len() {
                        return false;
                    }

                    pending.extend(
                        xs.iter()
                            .zip(ys.iter())
                            .rev()
                            .map(|(x, y)| (x.as_ref(), y.as_ref())),
                    );
                }
                (Constant::ProtoPair(f1, s1, l1, r1), Constant::ProtoPair(f2, s2, l2, r2)) => {
                    if !types_equal(f1, f2, &mut seen_types)
                        || !types_equal(s1, s2, &mut seen_types)
                    {
                        return false;
                    }

                    pending.push((r1, r2));
                    pending.push((l1, l2));
                }
                (Constant::Integer(x), Constant::Integer(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::ByteString(x), Constant::ByteString(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::String(x), Constant::String(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::Unit, Constant::Unit) => {}
                (Constant::Bool(x), Constant::Bool(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::Data(x), Constant::Data(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::Bls12_381G1Element(x), Constant::Bls12_381G1Element(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::Bls12_381G2Element(x), Constant::Bls12_381G2Element(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::Bls12_381MlResult(x), Constant::Bls12_381MlResult(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Constant::Value(x), Constant::Value(y)) => {
                    if x != y {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        true
    }
}

/// A Name containing it's parsed textual representation
/// and a unique id from string interning. The Name's text is
/// interned during parsing.
#[derive(Debug, Clone, Eq)]
pub struct Name {
    pub text: String,
    pub unique: Unique,
}

impl Name {
    pub fn text(t: impl ToString) -> Name {
        Name {
            text: t.to_string(),
            unique: 0.into(),
        }
    }
}

impl hash::Hash for Name {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        self.unique.hash(state);
    }
}

impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        self.unique == other.unique && self.text == other.text
    }
}

/// A unique id used for string interning.
#[derive(Debug, Clone, PartialEq, Copy, Eq, Hash)]
pub struct Unique(isize);

impl Unique {
    /// Create a new unique id.
    pub fn new(unique: isize) -> Self {
        Unique(unique)
    }

    /// Increment the available unique id. This is used during
    /// string interning to get the next available unique id.
    pub fn increment(&mut self) {
        self.0 += 1;
    }
}

impl From<isize> for Unique {
    fn from(i: isize) -> Self {
        Unique(i)
    }
}

impl From<Unique> for isize {
    fn from(d: Unique) -> Self {
        d.0
    }
}

impl Display for Unique {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Similar to `Name` but for Debruijn indices.
/// `Name` is replaced by `NamedDebruijn` when converting
/// program to it's debruijn form.
#[derive(Debug, Clone, Eq)]
pub struct NamedDeBruijn {
    pub text: String,
    pub index: DeBruijn,
}

impl PartialEq for NamedDeBruijn {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

/// This is useful for decoding a on chain program into debruijn form.
/// It allows for injecting fake textual names while also using Debruijn for decoding
/// without having to loop through twice.
#[derive(Debug, Clone)]
pub struct FakeNamedDeBruijn(pub(crate) NamedDeBruijn);

impl From<DeBruijn> for FakeNamedDeBruijn {
    fn from(d: DeBruijn) -> Self {
        FakeNamedDeBruijn(d.into())
    }
}

impl From<FakeNamedDeBruijn> for DeBruijn {
    fn from(d: FakeNamedDeBruijn) -> Self {
        d.0.into()
    }
}

impl From<FakeNamedDeBruijn> for NamedDeBruijn {
    fn from(d: FakeNamedDeBruijn) -> Self {
        d.0
    }
}

impl From<NamedDeBruijn> for FakeNamedDeBruijn {
    fn from(d: NamedDeBruijn) -> Self {
        FakeNamedDeBruijn(d)
    }
}

/// Represents a debruijn index.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct DeBruijn(usize);

impl DeBruijn {
    /// Create a new debruijn index.
    pub fn new(index: usize) -> Self {
        DeBruijn(index)
    }

    pub fn inner(&self) -> usize {
        self.0
    }
}

impl From<usize> for DeBruijn {
    fn from(i: usize) -> Self {
        DeBruijn(i)
    }
}

impl From<DeBruijn> for usize {
    fn from(d: DeBruijn) -> Self {
        d.0
    }
}

impl Display for DeBruijn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NamedDeBruijn> for DeBruijn {
    fn from(n: NamedDeBruijn) -> Self {
        n.index
    }
}

impl From<DeBruijn> for NamedDeBruijn {
    fn from(index: DeBruijn) -> Self {
        NamedDeBruijn {
            // Inject fake name. We got `i` from the Plutus code base.
            text: String::from("i"),
            index,
        }
    }
}

/// Convert a Parsed `Program` to a `Program` in `NamedDebruijn` form.
/// This checks for any Free Uniques in the `Program` and returns an error if found.
impl TryFrom<Program<Name>> for Program<NamedDeBruijn> {
    type Error = debruijn::Error;

    fn try_from(value: Program<Name>) -> Result<Self, Self::Error> {
        Ok(Program::<NamedDeBruijn> {
            version: value.version,
            term: value.term.try_into()?,
        })
    }
}

/// Convert a Parsed `Term` to a `Term` in `NamedDebruijn` form.
/// This checks for any Free Uniques in the `Term` and returns an error if found.
impl TryFrom<Term<Name>> for Term<NamedDeBruijn> {
    type Error = debruijn::Error;

    fn try_from(value: Term<Name>) -> Result<Self, debruijn::Error> {
        let mut converter = Converter::new();

        let term = converter.name_to_named_debruijn(&value)?;

        Ok(term)
    }
}

/// Convert a Parsed `Program` to a `Program` in `Debruijn` form.
/// This checks for any Free Uniques in the `Program` and returns an error if found.
impl TryFrom<Program<Name>> for Program<DeBruijn> {
    type Error = debruijn::Error;

    fn try_from(value: Program<Name>) -> Result<Self, Self::Error> {
        Ok(Program::<DeBruijn> {
            version: value.version,
            term: value.term.try_into()?,
        })
    }
}

/// Convert a Parsed `Term` to a `Term` in `Debruijn` form.
/// This checks for any Free Uniques in the `Program` and returns an error if found.
impl TryFrom<Term<Name>> for Term<DeBruijn> {
    type Error = debruijn::Error;

    fn try_from(value: Term<Name>) -> Result<Self, debruijn::Error> {
        let mut converter = Converter::new();

        let term = converter.name_to_debruijn(&value)?;

        Ok(term)
    }
}

impl TryFrom<&Program<DeBruijn>> for Program<Name> {
    type Error = debruijn::Error;

    fn try_from(value: &Program<DeBruijn>) -> Result<Self, Self::Error> {
        Ok(Program::<Name> {
            version: value.version,
            term: (&value.term).try_into()?,
        })
    }
}

impl TryFrom<&Term<DeBruijn>> for Term<Name> {
    type Error = debruijn::Error;

    fn try_from(value: &Term<DeBruijn>) -> Result<Self, debruijn::Error> {
        let mut converter = Converter::new();

        let term = converter.debruijn_to_name(value)?;

        Ok(term)
    }
}

impl TryFrom<Program<NamedDeBruijn>> for Program<Name> {
    type Error = debruijn::Error;

    fn try_from(value: Program<NamedDeBruijn>) -> Result<Self, Self::Error> {
        Ok(Program::<Name> {
            version: value.version,
            term: value.term.try_into()?,
        })
    }
}

impl TryFrom<Term<NamedDeBruijn>> for Term<Name> {
    type Error = debruijn::Error;

    fn try_from(value: Term<NamedDeBruijn>) -> Result<Self, debruijn::Error> {
        let mut converter = Converter::new();

        let term = converter.named_debruijn_to_name(&value)?;

        Ok(term)
    }
}

impl From<Program<NamedDeBruijn>> for Program<DeBruijn> {
    fn from(value: Program<NamedDeBruijn>) -> Self {
        Program::<DeBruijn> {
            version: value.version,
            term: value.term.into(),
        }
    }
}

impl From<Term<NamedDeBruijn>> for Term<DeBruijn> {
    fn from(value: Term<NamedDeBruijn>) -> Self {
        let mut converter = Converter::new();

        converter.named_debruijn_to_debruijn(&value)
    }
}

impl From<Program<NamedDeBruijn>> for Program<FakeNamedDeBruijn> {
    fn from(value: Program<NamedDeBruijn>) -> Self {
        Program::<FakeNamedDeBruijn> {
            version: value.version,
            term: value.term.into(),
        }
    }
}

impl From<Term<NamedDeBruijn>> for Term<FakeNamedDeBruijn> {
    fn from(value: Term<NamedDeBruijn>) -> Self {
        let mut converter = Converter::new();

        converter.named_debruijn_to_fake_named_debruijn(&value)
    }
}

impl TryFrom<Program<DeBruijn>> for Program<Name> {
    type Error = debruijn::Error;

    fn try_from(value: Program<DeBruijn>) -> Result<Self, Self::Error> {
        Ok(Program::<Name> {
            version: value.version,
            term: value.term.try_into()?,
        })
    }
}

impl TryFrom<Term<DeBruijn>> for Term<Name> {
    type Error = debruijn::Error;

    fn try_from(value: Term<DeBruijn>) -> Result<Self, debruijn::Error> {
        let mut converter = Converter::new();

        let term = converter.debruijn_to_name(&value)?;

        Ok(term)
    }
}

impl From<Program<DeBruijn>> for Program<NamedDeBruijn> {
    fn from(value: Program<DeBruijn>) -> Self {
        Program::<NamedDeBruijn> {
            version: value.version,
            term: value.term.into(),
        }
    }
}

impl From<Term<DeBruijn>> for Term<NamedDeBruijn> {
    fn from(value: Term<DeBruijn>) -> Self {
        let mut converter = Converter::new();

        converter.debruijn_to_named_debruijn(&value)
    }
}

impl From<Program<FakeNamedDeBruijn>> for Program<NamedDeBruijn> {
    fn from(value: Program<FakeNamedDeBruijn>) -> Self {
        Program::<NamedDeBruijn> {
            version: value.version,
            term: value.term.into(),
        }
    }
}

impl From<Term<FakeNamedDeBruijn>> for Term<NamedDeBruijn> {
    fn from(value: Term<FakeNamedDeBruijn>) -> Self {
        let mut converter = Converter::new();

        converter.fake_named_debruijn_to_named_debruijn(&value)
    }
}

impl Program<NamedDeBruijn> {
    pub fn eval(self, initial_budget: ExBudget) -> EvalResult {
        let mut machine = Machine::new(
            Language::PlutusV3,
            CostModel::default(),
            initial_budget,
            200,
        );

        let term = machine.run(self.term);

        EvalResult::new(
            term,
            machine.ex_budget,
            initial_budget,
            machine.traces,
            machine.spend_counter.map(|i| i.into()),
        )
    }

    /// Evaluate a Program as a specific PlutusVersion
    pub fn eval_version(self, initial_budget: ExBudget, version: &Language) -> EvalResult {
        let mut machine = Machine::new(version.clone(), CostModel::default(), initial_budget, 200);

        let term = machine.run(self.term);

        EvalResult::new(
            term,
            machine.ex_budget,
            initial_budget,
            machine.traces,
            machine.spend_counter.map(|i| i.into()),
        )
    }

    /// Evaluate a Program as a specific PlutusVersion and protocol version,
    /// using the local protocol-aware default cost model for the chosen
    /// PlutusVersion.
    pub fn eval_version_with_protocol(
        self,
        initial_budget: ExBudget,
        version: &Language,
        protocol_major_version: u16,
    ) -> EvalResult {
        let mut machine = Machine::new_with_protocol(
            version.clone(),
            protocol_major_version,
            CostModel::default_for_language_and_protocol(version, protocol_major_version),
            initial_budget,
            200,
        );

        let term = machine.run(self.term);

        EvalResult::new(
            term,
            machine.ex_budget,
            initial_budget,
            machine.traces,
            machine.spend_counter.map(|i| i.into()),
        )
    }

    pub fn eval_as(
        self,
        version: &Language,
        costs: &[i64],
        initial_budget: Option<&ExBudget>,
    ) -> EvalResult {
        let budget = initial_budget.copied().unwrap_or_default();

        let mut machine = Machine::new(
            version.clone(),
            initialize_cost_model(version, costs),
            budget,
            200, //slippage
        );

        let term = machine.run(self.term);

        EvalResult::new(
            term,
            machine.ex_budget,
            budget,
            machine.traces,
            machine.spend_counter.map(|i| i.into()),
        )
    }

    /// Evaluate a Program with an explicit ledger cost model and protocol
    /// version.
    pub fn eval_as_with_protocol(
        self,
        version: &Language,
        protocol_major_version: u16,
        costs: &[i64],
        initial_budget: Option<&ExBudget>,
    ) -> EvalResult {
        let budget = initial_budget.copied().unwrap_or_default();

        let mut machine = Machine::new_with_protocol(
            version.clone(),
            protocol_major_version,
            initialize_cost_model_with_protocol(version, protocol_major_version, costs),
            budget,
            200, //slippage
        );

        let term = machine.run(self.term);

        EvalResult::new(
            term,
            machine.ex_budget,
            budget,
            machine.traces,
            machine.spend_counter.map(|i| i.into()),
        )
    }

    pub fn eval_debug(self, initial_budget: ExBudget, version: &Language) -> EvalResult {
        let mut machine = Machine::new_debug(
            version.clone(),
            CostModel::default(),
            initial_budget,
            200, //slippage
        );

        let term = machine.run(self.term);

        EvalResult::new(
            term,
            machine.ex_budget,
            initial_budget,
            machine.traces,
            machine.spend_counter.map(|i| i.into()),
        )
    }
}

impl Program<DeBruijn> {
    pub fn eval(&self, initial_budget: ExBudget) -> EvalResult {
        let program: Program<NamedDeBruijn> = self.clone().into();
        program.eval(initial_budget)
    }

    pub fn eval_version(self, initial_budget: ExBudget, version: &Language) -> EvalResult {
        let program: Program<NamedDeBruijn> = self.clone().into();
        program.eval_version(initial_budget, version)
    }

    pub fn eval_version_with_protocol(
        self,
        initial_budget: ExBudget,
        version: &Language,
        protocol_major_version: u16,
    ) -> EvalResult {
        let program: Program<NamedDeBruijn> = self.clone().into();
        program.eval_version_with_protocol(initial_budget, version, protocol_major_version)
    }
}

impl Term<NamedDeBruijn> {
    pub fn is_valid_script_result(&self) -> bool {
        !matches!(self, Term::Error { .. })
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{Constant, Data, Name, Term, Type, Value, ValueEntries, ValueError};
    use crate::builtins::DefaultFunction;
    use num_bigint::{BigInt, Sign};
    use pallas_codec::minicbor;
    use pallas_primitives::{alonzo::PlutusData, conway};
    use proptest::prelude::*;
    use std::{collections::BTreeMap, rc::Rc};

    /// Deep enough that a destructor taking one stack frame per level cannot survive the test
    /// harness's ~2 MiB thread stack, let alone the ~1 MiB a `wasm32` engine gives us. A script
    /// inside the on-chain size limit can nest far deeper than this: the cheapest level in flat is
    /// a 4-bit `delay` tag, half a byte.
    const DEEP: usize = 100_000;

    // Data's negative integers are encoded with an offset of 1, as an unsigned payload. This is unlike
    // num_bigint's BigInt; so both types representations aren't quite compatible with one another.
    #[test]
    fn integer_bigint_negative() {
        let large_negative_num: BigInt = BigInt::from(i128::MIN) - 1;

        let mut buf = vec![];
        minicbor::encode(Data::integer(large_negative_num.clone()), &mut buf)
            .expect("failed to encode bigint to CBOR");

        // NOTE: [2..] removes the CBOR tag and bytes len declaration.
        let large_negative_num_decoded = BigInt::from_bytes_be(Sign::Plus, &buf[2..]);

        assert_eq!(large_negative_num_decoded, -1 - large_negative_num);
    }

    /// Every spine below is built with a loop, never a recursive helper: a recursive builder would
    /// overflow while setting the test up and hide the thing we are actually testing.
    fn deep_delay_spine(depth: usize) -> Term<Name> {
        let mut term = Term::Error { uniq_id: 0 };

        for level in 0..depth {
            term = Term::Delay {
                body: Rc::new(term),
                uniq_id: level as isize,
            };
        }

        term
    }

    #[test]
    fn deep_term_drops_without_overflowing_the_stack() {
        drop(deep_delay_spine(DEEP));
    }

    #[test]
    fn deep_apply_spine_drops_without_overflowing_the_stack() {
        // `Apply` nests through two children at once, so this also covers a node being dismantled
        // in more than one direction.
        let mut term: Term<Name> = Term::Error { uniq_id: 0 };

        for level in 0..DEEP {
            term = Term::Apply {
                function: Rc::new(term),
                argument: Rc::new(Term::Error { uniq_id: 0 }),
                uniq_id: level as isize,
            };
        }

        drop(term);
    }

    #[test]
    fn deep_constr_and_case_spine_drops_without_overflowing_the_stack() {
        // `Constr` and `Case` nest through owned vectors rather than `Rc`s.
        let mut term: Term<Name> = Term::Error { uniq_id: 0 };

        for level in 0..DEEP {
            term = if level % 2 == 0 {
                Term::Constr {
                    tag: 0,
                    fields: vec![term],
                    uniq_id: level as isize,
                }
            } else {
                Term::Case {
                    constr: Rc::new(Term::Error { uniq_id: 0 }),
                    branches: vec![term],
                    uniq_id: level as isize,
                }
            };
        }

        drop(term);
    }

    #[test]
    fn dropping_a_term_leaves_shared_subterms_alone() {
        let shared = Rc::new(Term::Builtin {
            fun: DefaultFunction::AddInteger,
            uniq_id: 7,
        });

        let term: Term<Name> = Term::Apply {
            function: Rc::new(Term::Delay {
                body: Rc::clone(&shared),
                uniq_id: 1,
            }),
            argument: Rc::clone(&shared),
            uniq_id: 2,
        };

        assert_eq!(Rc::strong_count(&shared), 3);

        drop(term);

        // The teardown released both handles and left the pointee alone.
        assert_eq!(Rc::strong_count(&shared), 1);
        assert!(matches!(
            shared.as_ref(),
            Term::Builtin {
                fun: DefaultFunction::AddInteger,
                uniq_id: 7,
            }
        ));
    }

    #[test]
    fn dropping_a_term_leaves_a_shared_deep_subterm_intact() {
        let spine = Rc::new(deep_delay_spine(DEEP));

        let first = Term::Force {
            body: Rc::clone(&spine),
            uniq_id: -1,
        };

        let second = Term::Force {
            body: Rc::clone(&spine),
            uniq_id: -2,
        };

        drop(first);

        // Walking (iteratively, again) still finds every level: the shared spine was not
        // dismantled out from under the other holders.
        let mut depth = 0;
        let mut cursor = spine.as_ref();

        while let Term::Delay { body, .. } = cursor {
            depth += 1;
            cursor = body.as_ref();
        }

        assert_eq!(depth, DEEP);
        assert_eq!(Rc::strong_count(&spine), 2);

        drop(second);
        drop(spine);
    }

    #[test]
    fn deep_constant_drops_without_overflowing_the_stack() {
        // The element type is deliberately left flat here: `Type` derives `Clone`, which is itself
        // recursive, so a matching nested type would overflow while the test was still setting up.
        // Deep types get their own test below.
        let mut constant = Constant::Integer(BigInt::from(1));

        for _ in 0..DEEP {
            constant = Constant::ProtoList(Type::Integer, vec![Rc::new(constant)]);
        }

        drop(constant);

        let mut constant = Constant::Unit;

        for _ in 0..DEEP {
            constant = Constant::ProtoPair(
                Type::Unit,
                Type::Unit,
                Rc::new(constant),
                Rc::new(Constant::Unit),
            );
        }

        drop(constant);
    }

    #[test]
    fn deep_type_drops_without_overflowing_the_stack() {
        let mut typ = Type::Integer;

        for _ in 0..DEEP {
            typ = Type::List(Rc::new(typ));
        }

        drop(typ);
    }

    /// A `constr` / `case` spine, which nests through owned vectors rather than `Rc`s.
    fn deep_constr_spine(depth: usize) -> Term<Name> {
        let mut term: Term<Name> = Term::Error { uniq_id: 0 };

        for level in 0..depth {
            term = if level % 2 == 0 {
                Term::Constr {
                    tag: level,
                    fields: vec![term],
                    uniq_id: level as isize,
                }
            } else {
                Term::Case {
                    constr: Rc::new(Term::Error { uniq_id: 0 }),
                    branches: vec![Term::Error { uniq_id: 0 }, term],
                    uniq_id: level as isize,
                }
            };
        }

        term
    }

    #[test]
    fn deep_terms_clone_and_compare_without_overflowing_the_stack() {
        for term in [deep_constr_spine(DEEP), deep_delay_spine(DEEP)] {
            let copy = term.clone();
            assert!(copy == term);

            let deep_copy = term.deep_clone();
            assert!(deep_copy == term);
        }

        // A difference at the very bottom is still found.
        let left = deep_constr_spine(DEEP);
        let mut right: Term<Name> = Term::Builtin {
            fun: DefaultFunction::AddInteger,
            uniq_id: 0,
        };
        for level in 0..DEEP {
            right = if level % 2 == 0 {
                Term::Constr {
                    tag: level,
                    fields: vec![right],
                    uniq_id: -1,
                }
            } else {
                Term::Case {
                    constr: Rc::new(Term::Error { uniq_id: -1 }),
                    branches: vec![Term::Error { uniq_id: -1 }, right],
                    uniq_id: -1,
                }
            };
        }
        assert!(left != right);
    }

    #[test]
    fn term_clone_keeps_ids_and_shares_rc_children() {
        let shared = Rc::new(Term::<Name>::Error { uniq_id: 9 });
        let term = Term::Constr {
            tag: 3,
            fields: vec![
                Term::Delay {
                    body: Rc::clone(&shared),
                    uniq_id: 1,
                },
                Term::Case {
                    constr: Rc::clone(&shared),
                    branches: vec![Term::Error { uniq_id: 2 }],
                    uniq_id: 3,
                },
            ],
            uniq_id: 4,
        };

        let copy = term.clone();

        assert!(copy == term);
        assert_eq!(copy.uniq_id(), 4);
        assert_eq!(Rc::strong_count(&shared), 5);

        let Term::Constr { fields, .. } = &copy else {
            panic!("not a constr")
        };
        assert_eq!(fields[0].uniq_id(), 1);
        assert_eq!(fields[1].uniq_id(), 3);
    }

    #[test]
    fn deep_wrappers_negate_and_count_as_constants() {
        let mut term: Term<Name> = Term::Constant {
            value: Constant::Integer(BigInt::from(7)).into(),
            uniq_id: 0,
        };
        for level in 0..DEEP {
            term = if level % 2 == 0 {
                Term::Delay {
                    body: Rc::new(term),
                    uniq_id: 0,
                }
            } else {
                Term::Force {
                    body: Rc::new(term),
                    uniq_id: 0,
                }
            };
        }

        assert!(term.is_constant());

        let negated = term
            .try_negate()
            .expect("an integer under delays and forces");
        let mut cursor = &negated;
        let mut wrappers = 0;
        while let Term::Delay { body, .. } | Term::Force { body, .. } = cursor {
            wrappers += 1;
            cursor = body;
        }
        assert_eq!(wrappers, DEEP);
        assert!(cursor.is_int());
        assert!(
            matches!(cursor, Term::Constant { value, .. } if value.as_ref() == &Constant::Integer(BigInt::from(-7)))
        );
        // Fresh ids, handed out innermost first.
        assert!(negated.uniq_id() > cursor.uniq_id());
    }

    /// A list nested `depth` times, its declared type as deep, around `leaf`.
    fn deep_list(depth: usize, leaf: Constant, leaf_type: Type) -> Constant {
        let mut constant = leaf;
        let mut typ = leaf_type;

        for _ in 0..depth {
            constant = Constant::ProtoList(typ.clone(), vec![Rc::new(constant)]);
            typ = Type::List(Rc::new(typ));
        }

        constant
    }

    /// A pair nested `depth` times through its first component.
    fn deep_pair(depth: usize, leaf: Constant, leaf_type: Type) -> Constant {
        let mut constant = leaf;
        let mut typ = leaf_type;

        for _ in 0..depth {
            constant = Constant::ProtoPair(
                typ.clone(),
                Type::Bool,
                Rc::new(constant),
                Rc::new(Constant::Bool(true)),
            );
            typ = Type::Pair(Rc::new(typ), Rc::new(Type::Bool));
        }

        constant
    }

    #[test]
    fn deep_constants_compare_clone_and_check_without_overflowing_the_stack() {
        for build in [deep_list, deep_pair] {
            let constant = Rc::new(build(
                DEEP,
                Constant::Integer(BigInt::from(1)),
                Type::Integer,
            ));

            assert!(*constant == build(DEEP, Constant::Integer(BigInt::from(1)), Type::Integer));
            assert!(*constant != build(DEEP, Constant::Integer(BigInt::from(2)), Type::Integer));

            assert!(!Constant::contains_value(&constant));
            assert!(!Constant::contains_array(&constant));

            let with_value = Rc::new(build(DEEP, Constant::Value(Value::empty()), Type::Value));
            assert!(Constant::contains_value(&with_value));
            assert!(!Constant::contains_array(&with_value));

            let with_array = Rc::new(build(
                DEEP,
                Constant::ProtoArray(Type::Unit, vec![]),
                Type::Array(Rc::new(Type::Unit)),
            ));
            assert!(Constant::contains_array(&with_array));
            assert!(!Constant::contains_value(&with_array));
        }

        // `deep_clone` shares nothing, so it copies every level's declared type afresh; keep the
        // types flat here and let the nesting come from the values alone.
        let mut constant = Constant::Integer(BigInt::from(1));
        for level in 0..DEEP {
            constant = if level % 2 == 0 {
                Constant::ProtoList(Type::Integer, vec![Rc::new(constant)])
            } else {
                Constant::ProtoPair(
                    Type::Integer,
                    Type::Unit,
                    Rc::new(constant),
                    Rc::new(Constant::Unit),
                )
            };
        }
        assert!(constant.deep_clone() == constant);
    }

    #[test]
    fn deep_types_compare_print_and_check_without_overflowing_the_stack() {
        let mut typ = Type::Integer;
        let mut other = Type::Integer;

        for level in 0..DEEP {
            if level % 2 == 0 {
                typ = Type::List(Rc::new(typ));
                other = Type::List(Rc::new(other));
            } else {
                typ = Type::Pair(Rc::new(Type::Unit), Rc::new(typ));
                other = Type::Pair(Rc::new(Type::Unit), Rc::new(other));
            }
        }

        assert!(typ == other);
        assert!(typ.deep_clone() == typ);
        assert!(!typ.contains_value());
        assert!(!typ.contains_array());

        let text = typ.to_string();
        assert!(text.starts_with("pair unit list pair unit list "));
        assert!(text.ends_with(" list integer"));

        let small = Type::Pair(
            Rc::new(Type::List(Rc::new(Type::Array(Rc::new(Type::Value))))),
            Rc::new(Type::Bool),
        );
        assert_eq!(small.to_string(), "pair list array value bool");
        assert!(small.contains_value());
        assert!(small.contains_array());
    }

    #[test]
    fn availability_checks_visit_shared_nodes_once() {
        // Both components of every pair are the same `Rc`: 2^64 paths, 64 nodes.
        let mut constant = Rc::new(Constant::Unit);
        let mut typ = Rc::new(Type::Unit);

        for _ in 0..64 {
            constant = Rc::new(Constant::ProtoPair(
                typ.as_ref().clone(),
                typ.as_ref().clone(),
                Rc::clone(&constant),
                Rc::clone(&constant),
            ));
            typ = Rc::new(Type::Pair(Rc::clone(&typ), Rc::clone(&typ)));
        }

        assert!(!Constant::contains_value(&constant));
        assert!(!Constant::contains_array(&constant));
        assert!(!typ.contains_value());
    }

    fn data_value_with_quantity(quantity: conway::BigInt) -> PlutusData {
        Data::map(vec![(
            Data::bytestring(vec![0]),
            Data::map(vec![(
                Data::bytestring(vec![0]),
                PlutusData::BigInt(quantity),
            )]),
        )])
    }

    #[test]
    fn insert_coin_updates_size_metadata_at_max_boundaries() {
        let original = Value::from_canonical_bounded_entries(vec![
            (vec![0], vec![(vec![0], -1), (vec![1], 1), (vec![2], 1)]),
            (vec![1], vec![(vec![0], 2), (vec![1], 3)]),
            (vec![2], vec![(vec![0], -4)]),
        ])
        .unwrap();
        assert_eq!(original.total_size(), 6);
        assert_eq!(original.outer_size(), 3);
        assert_eq!(original.max_inner_size(), 3);
        assert_eq!(original.negative_amounts(), 2);

        let largest_shrunk = original.insert_coin(&[0], &[0], &BigInt::from(0)).unwrap();
        assert_eq!(largest_shrunk.total_size(), 5);
        assert_eq!(largest_shrunk.max_inner_size(), 2);
        assert_eq!(largest_shrunk.negative_amounts(), 1);

        let quantity_overwritten = largest_shrunk
            .insert_coin(&[1], &[0], &BigInt::from(-2))
            .unwrap();
        assert_eq!(quantity_overwritten.total_size(), 5);
        assert_eq!(quantity_overwritten.max_inner_size(), 2);
        assert_eq!(quantity_overwritten.negative_amounts(), 2);

        let inner_removed = quantity_overwritten
            .insert_coin(&[2], &[0], &BigInt::from(0))
            .unwrap();
        assert_eq!(inner_removed.total_size(), 4);
        assert_eq!(inner_removed.outer_size(), 2);
        assert_eq!(inner_removed.max_inner_size(), 2);
        assert_eq!(inner_removed.negative_amounts(), 1);

        let singleton =
            Value::from_canonical_bounded_entries(vec![(vec![0], vec![(vec![0], 1)])]).unwrap();
        let empty = singleton.insert_coin(&[0], &[0], &BigInt::from(0)).unwrap();
        assert_eq!(empty.total_size(), 0);
        assert_eq!(empty.outer_size(), 0);
        assert_eq!(empty.max_inner_size(), 0);
    }

    #[test]
    fn value_preserves_canonical_entries_and_data_roundtrip() {
        let entries = vec![
            (vec![], vec![(vec![], i128::MIN), (vec![0xff], i128::MAX)]),
            (vec![0xff; 32], vec![(vec![0; 32], -1)]),
        ];
        let value = Value::from_canonical_bounded_entries(entries.clone()).unwrap();

        assert_eq!(
            Value::from_data(&value.to_data_unchecked()),
            Ok(value.clone())
        );
        assert_eq!(value.into_entries(), entries);
    }

    #[test]
    fn from_data_rejects_oversized_keys_before_copying() {
        Value::reset_data_key_copy_count();
        let valid = data_value_with_quantity(conway::BigInt::BigUInt(vec![1].into()));
        Value::from_data(&valid).unwrap();
        assert_eq!(Value::data_key_copy_count(), 2);

        Value::reset_data_key_copy_count();
        let oversized_currency = vec![0xff; 256 * 1024];
        let data = Data::map(vec![(
            Data::bytestring(oversized_currency),
            Data::integer(BigInt::from(1)),
        )]);
        assert_eq!(
            Value::from_data(&data),
            Err(ValueError::KeyTooLong(256 * 1024))
        );
        assert_eq!(Value::data_key_copy_count(), 0);

        Value::reset_data_key_copy_count();
        let oversized_token = vec![0xff; 256 * 1024];
        let data = Data::map(vec![(
            Data::bytestring(vec![0]),
            Data::map(vec![(
                Data::bytestring(oversized_token),
                Data::bytestring(vec![0]),
            )]),
        )]);
        assert_eq!(
            Value::from_data(&data),
            Err(ValueError::KeyTooLong(256 * 1024))
        );
        assert_eq!(Value::data_key_copy_count(), 0);
    }

    #[test]
    fn from_data_accepts_leading_zero_pallas_bignums() {
        for (quantity, expected) in [
            (conway::BigInt::BigUInt(vec![0, 0, 1].into()), 1),
            (conway::BigInt::BigNInt(vec![0, 0, 1].into()), -2),
        ] {
            let data = data_value_with_quantity(quantity);
            assert_eq!(
                Value::from_data(&data).unwrap().lookup_coin(&[0], &[0]),
                expected
            );
        }
    }

    type ValueModel = BTreeMap<Vec<u8>, BTreeMap<Vec<u8>, i128>>;

    fn apply_model_update(model: &mut ValueModel, currency: u8, token: u8, quantity: i16) {
        let currency = vec![currency];
        let token = vec![token];
        if quantity == 0 {
            if let Some(tokens) = model.get_mut(&currency) {
                tokens.remove(&token);
                if tokens.is_empty() {
                    model.remove(&currency);
                }
            }
        } else {
            model
                .entry(currency)
                .or_default()
                .insert(token, i128::from(quantity));
        }
    }

    fn model_entries(model: &ValueModel) -> ValueEntries {
        model
            .iter()
            .map(|(currency, tokens)| {
                (
                    currency.clone(),
                    tokens
                        .iter()
                        .map(|(token, quantity)| (token.clone(), *quantity))
                        .collect(),
                )
            })
            .collect()
    }

    fn assert_value_matches_model(value: &Value, model: &ValueModel) {
        assert_eq!(value.clone().into_entries(), model_entries(model));
        assert_eq!(value.outer_size(), model.len());
        assert_eq!(
            value.total_size(),
            model.values().map(BTreeMap::len).sum::<usize>()
        );
        assert_eq!(
            value.max_inner_size(),
            model.values().map(BTreeMap::len).max().unwrap_or(0)
        );
        assert_eq!(
            value.negative_amounts(),
            model
                .values()
                .flat_map(BTreeMap::values)
                .filter(|quantity| quantity.is_negative())
                .count()
        );
        assert_eq!(
            Value::from_data(&value.to_data_unchecked()),
            Ok(value.clone())
        );
    }

    fn value_and_model(operations: &[(u8, u8, i16)]) -> (Value, ValueModel) {
        let mut value = Value::empty();
        let mut model = ValueModel::new();
        for (currency, token, quantity) in operations {
            value = value
                .insert_coin(&[*currency], &[*token], &BigInt::from(*quantity))
                .unwrap();
            apply_model_update(&mut model, *currency, *token, *quantity);
            assert_value_matches_model(&value, &model);
        }
        (value, model)
    }

    fn union_models(mut left: ValueModel, right: &ValueModel) -> ValueModel {
        for (currency, tokens) in right {
            for (token, quantity) in tokens {
                let combined = left
                    .get(currency)
                    .and_then(|tokens| tokens.get(token))
                    .copied()
                    .unwrap_or(0)
                    + quantity;
                if combined == 0 {
                    if let Some(tokens) = left.get_mut(currency) {
                        tokens.remove(token);
                        if tokens.is_empty() {
                            left.remove(currency);
                        }
                    }
                } else {
                    left.entry(currency.clone())
                        .or_default()
                        .insert(token.clone(), combined);
                }
            }
        }
        left
    }

    fn scale_model(model: &ValueModel, scalar: i16) -> ValueModel {
        if scalar == 0 {
            return ValueModel::new();
        }
        model
            .iter()
            .map(|(currency, tokens)| {
                (
                    currency.clone(),
                    tokens
                        .iter()
                        .map(|(token, quantity)| (token.clone(), quantity * i128::from(scalar)))
                        .collect(),
                )
            })
            .collect()
    }

    proptest! {
        #[test]
        fn value_operations_match_btree_model(
            left_operations in prop::collection::vec((any::<u8>(), any::<u8>(), -1000i16..=1000), 0..80),
            right_operations in prop::collection::vec((any::<u8>(), any::<u8>(), -1000i16..=1000), 0..80),
            scalar in -4i16..=4,
        ) {
            let (left, left_model) = value_and_model(&left_operations);
            let (right, right_model) = value_and_model(&right_operations);

            for (currency, tokens) in &left_model {
                for (token, quantity) in tokens {
                    prop_assert_eq!(left.lookup_coin(currency, token), *quantity);
                }
            }

            let union = left.union(&right).unwrap();
            let union_model = union_models(left_model.clone(), &right_model);
            assert_value_matches_model(&union, &union_model);

            let scaled = left.scale(&BigInt::from(scalar)).unwrap();
            assert_value_matches_model(&scaled, &scale_model(&left_model, scalar));

            let expected_contains = if left_model
                .values()
                .flat_map(BTreeMap::values)
                .any(|quantity| quantity.is_negative())
            {
                Err(ValueError::FirstValueContainsNegativeAmounts)
            } else if right_model
                .values()
                .flat_map(BTreeMap::values)
                .any(|quantity| quantity.is_negative())
            {
                Err(ValueError::SecondValueContainsNegativeAmounts)
            } else {
                Ok(right_model.iter().all(|(currency, tokens)| {
                    tokens.iter().all(|(token, quantity)| {
                        left_model
                            .get(currency)
                            .and_then(|tokens| tokens.get(token))
                            .copied()
                            .unwrap_or(0)
                            >= *quantity
                    })
                }))
            };
            prop_assert_eq!(left.contains(&right), expected_contains);
        }
    }
}
