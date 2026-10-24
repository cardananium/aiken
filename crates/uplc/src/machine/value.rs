use super::{
    Error,
    runtime::{self, BuiltinRuntime, BuiltinSemantics},
};
use crate::{
    ast::{Constant, NamedDeBruijn, Term, Type},
    builtins::DefaultFunction,
    debug::{self, Item},
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use pallas_primitives::conway::{self, PlutusData};
use std::{collections::VecDeque, fmt, mem, mem::size_of, ops::Deref, rc::Rc};

pub type Env = Rc<Vec<Value>>;

pub enum Value {
    Con(Rc<Constant>),
    Delay {
        body: Rc<Term<NamedDeBruijn>>,
        env: Env,
        term_id: isize,
    },
    Lambda {
        parameter_name: Rc<NamedDeBruijn>,
        body: Rc<Term<NamedDeBruijn>>,
        env: Env,
        term_id: isize,
    },
    Builtin {
        fun: DefaultFunction,
        runtime: BuiltinRuntime,
        term_id: isize,
    },
    Constr {
        tag: usize,
        fields: Vec<Value>,
        term_id: isize,
    },
}

/// `Value` nests two ways — through `Constr` fields and through the captured `Env` of a closure,
/// which is a vector of further values — and both are as deep as the script makes them: every
/// nested lambda the machine enters extends the environment chain. As with `Term` (see the `Drop`
/// impl in `ast.rs` for the full rationale) the derived destructor would walk that chain one stack
/// frame at a time, which is not survivable on `wasm32`. Do not remove this as redundant.
impl Drop for Value {
    fn drop(&mut self) {
        let mut pending: Vec<Value> = Vec::new();
        let mut filler: Option<Env> = None;

        detach_value_children(self, &mut pending, &mut filler);

        while let Some(mut value) = pending.pop() {
            detach_value_children(&mut value, &mut pending, &mut filler);
        }
    }
}

/// Move the values `value` solely owns onto `pending`, leaving it childless.
///
/// The `Rc<Term>` and `Rc<Constant>` a value holds are left alone: those carry their own iterative
/// `Drop`, so releasing them here does not recurse either.
fn detach_value_children(value: &mut Value, pending: &mut Vec<Value>, filler: &mut Option<Env>) {
    match value {
        Value::Con(_) => {}

        Value::Delay { env, .. } | Value::Lambda { env, .. } => {
            detach_env(env, pending, filler);
        }

        Value::Builtin { runtime, .. } => {
            // Partially applied arguments are owned outright.
            pending.append(&mut runtime.args);
        }

        Value::Constr { fields, .. } => {
            pending.append(fields);
        }
    }
}

/// Take the environment behind `slot`, but only when we are its sole owner.
///
/// Environments are shared by every closure captured under them, so an `Env` whose strong count is
/// above one is not ours to empty — other closures still read it. That is also the exact condition
/// under which `Rc::try_unwrap` refuses, and dropping a shared `Rc` alongside its value is O(1) and
/// never touches the pointee. When we do own it we swap in a leaf first; that leaf is a clone of a
/// single empty environment shared by the whole teardown, so the shallow `Drop` that later visits
/// the vacated slot sees a strong count above one and skips it.
fn detach_env(slot: &mut Env, pending: &mut Vec<Value>, filler: &mut Option<Env>) {
    if Rc::strong_count(slot) != 1 {
        return;
    }

    let filler = filler.get_or_insert_with(|| Rc::new(Vec::new()));

    if let Ok(mut env) = Rc::try_unwrap(mem::replace(slot, Rc::clone(filler))) {
        pending.append(&mut env);
    }
}

/// The derived format, `{:?}` and `{:#?}` alike, written without recursion (see `debug.rs`):
/// values nest through `Constr` fields, builtin arguments and closure environments.
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn values(values: &[Value]) -> Item<'_, Value> {
            Item::List(values.iter().map(Item::Node).collect())
        }

        debug::write(self, f, |value| match value {
            Value::Con(constant) => Item::Tuple("Con", vec![Item::Leaf(constant)]),
            Value::Delay { body, env, term_id } => Item::Struct(
                "Delay",
                vec![
                    ("body", Item::Leaf(body)),
                    ("env", values(env)),
                    ("term_id", Item::Leaf(term_id)),
                ],
            ),
            Value::Lambda {
                parameter_name,
                body,
                env,
                term_id,
            } => Item::Struct(
                "Lambda",
                vec![
                    ("parameter_name", Item::Leaf(parameter_name)),
                    ("body", Item::Leaf(body)),
                    ("env", values(env)),
                    ("term_id", Item::Leaf(term_id)),
                ],
            ),
            Value::Builtin {
                fun,
                runtime,
                term_id,
            } => Item::Struct(
                "Builtin",
                vec![
                    ("fun", Item::Leaf(fun)),
                    (
                        "runtime",
                        Item::Struct(
                            "BuiltinRuntime",
                            vec![
                                ("args", values(&runtime.args)),
                                ("fun", Item::Leaf(&runtime.fun)),
                                ("forces", Item::Leaf(&runtime.forces)),
                            ],
                        ),
                    ),
                    ("term_id", Item::Leaf(term_id)),
                ],
            ),
            Value::Constr {
                tag,
                fields,
                term_id,
            } => Item::Struct(
                "Constr",
                vec![
                    ("tag", Item::Leaf(tag)),
                    ("fields", values(fields)),
                    ("term_id", Item::Leaf(term_id)),
                ],
            ),
        })
    }
}

/// A value's children are `Rc`s that clone in O(1) -- its constant, its body, its environment --
/// except a `Constr`'s fields and a partially applied builtin's arguments, which are owned
/// outright and nest as deep as the script makes them. The derived `Clone` copied those one stack
/// frame per level; they are copied off an explicit stack instead, and every other value is copied
/// in place with no allocation, which is what `lookup_var` does on every variable.
impl Clone for Value {
    fn clone(&self) -> Self {
        fn shallow(value: &Value) -> Option<Value> {
            Some(match value {
                Value::Con(constant) => Value::Con(Rc::clone(constant)),
                Value::Delay { body, env, term_id } => Value::Delay {
                    body: Rc::clone(body),
                    env: Rc::clone(env),
                    term_id: *term_id,
                },
                Value::Lambda {
                    parameter_name,
                    body,
                    env,
                    term_id,
                } => Value::Lambda {
                    parameter_name: Rc::clone(parameter_name),
                    body: Rc::clone(body),
                    env: Rc::clone(env),
                    term_id: *term_id,
                },
                Value::Builtin { runtime, .. } if !runtime.args.is_empty() => return None,
                Value::Builtin {
                    fun,
                    runtime,
                    term_id,
                } => Value::Builtin {
                    fun: *fun,
                    runtime: BuiltinRuntime {
                        args: Vec::new(),
                        fun: runtime.fun,
                        forces: runtime.forces,
                    },
                    term_id: *term_id,
                },
                Value::Constr { fields, .. } if !fields.is_empty() => return None,
                Value::Constr { tag, term_id, .. } => Value::Constr {
                    tag: *tag,
                    fields: Vec::new(),
                    term_id: *term_id,
                },
            })
        }

        if let Some(copy) = shallow(self) {
            return copy;
        }

        enum Task<'a> {
            Copy(&'a Value),
            /// Rebuild this `Constr` or `Builtin` around the last copies made, its children.
            Assemble(&'a Value),
        }

        let mut tasks = vec![Task::Copy(self)];
        let mut copies: Vec<Value> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Copy(value) => match shallow(value) {
                    Some(copy) => copies.push(copy),
                    None => {
                        tasks.push(Task::Assemble(value));

                        let children = match value {
                            Value::Constr { fields, .. } => fields,
                            Value::Builtin { runtime, .. } => &runtime.args,
                            _ => unreachable!("only constr and builtin values are not shallow"),
                        };

                        tasks.extend(children.iter().rev().map(Task::Copy));
                    }
                },
                Task::Assemble(value) => {
                    let copy = match value {
                        Value::Constr {
                            tag,
                            fields,
                            term_id,
                        } => Value::Constr {
                            tag: *tag,
                            fields: copies.split_off(copies.len() - fields.len()),
                            term_id: *term_id,
                        },
                        Value::Builtin {
                            fun,
                            runtime,
                            term_id,
                        } => Value::Builtin {
                            fun: *fun,
                            runtime: BuiltinRuntime {
                                args: copies.split_off(copies.len() - runtime.args.len()),
                                fun: runtime.fun,
                                forces: runtime.forces,
                            },
                            term_id: *term_id,
                        },
                        _ => unreachable!("only constr and builtin values are assembled"),
                    };

                    copies.push(copy);
                }
            }
        }

        copies.pop().expect("the copy of the root")
    }
}

/// Structural equality, as derived, but iterative: values nest through `Constr` fields, builtin
/// arguments and -- for closures -- environments, all as deep as the script makes them (see the
/// `Drop` impl above). `pending` holds what is still to compare, in the order the derived
/// comparison visits it: a field compared after the children of the field before it waits there
/// as a `Check` holding its outcome.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        enum Pending<'a> {
            Values(&'a Value, &'a Value),
            Check(bool),
        }

        /// Queue `xs == ys` (lengths, then items in order), followed by `then`.
        fn queue<'a>(pending: &mut Vec<Pending<'a>>, xs: &'a [Value], ys: &'a [Value], then: bool) {
            pending.push(Pending::Check(then));
            pending.extend(
                xs.iter()
                    .zip(ys.iter())
                    .rev()
                    .map(|(x, y)| Pending::Values(x, y)),
            );
        }

        let mut pending = vec![Pending::Values(self, other)];

        while let Some(next) = pending.pop() {
            let pair = match next {
                Pending::Check(outcome) => {
                    if !outcome {
                        return false;
                    }
                    continue;
                }
                Pending::Values(left, right) => (left, right),
            };

            match pair {
                (Value::Con(c1), Value::Con(c2)) => {
                    if c1 != c2 {
                        return false;
                    }
                }
                (
                    Value::Delay {
                        body: b1,
                        env: e1,
                        term_id: i1,
                    },
                    Value::Delay {
                        body: b2,
                        env: e2,
                        term_id: i2,
                    },
                ) => {
                    if b1 != b2 || e1.len() != e2.len() {
                        return false;
                    }

                    queue(&mut pending, e1, e2, i1 == i2);
                }
                (
                    Value::Lambda {
                        parameter_name: p1,
                        body: b1,
                        env: e1,
                        term_id: i1,
                    },
                    Value::Lambda {
                        parameter_name: p2,
                        body: b2,
                        env: e2,
                        term_id: i2,
                    },
                ) => {
                    if p1 != p2 || b1 != b2 || e1.len() != e2.len() {
                        return false;
                    }

                    queue(&mut pending, e1, e2, i1 == i2);
                }
                (
                    Value::Builtin {
                        fun: f1,
                        runtime: r1,
                        term_id: i1,
                    },
                    Value::Builtin {
                        fun: f2,
                        runtime: r2,
                        term_id: i2,
                    },
                ) => {
                    if f1 != f2 || r1.args.len() != r2.args.len() {
                        return false;
                    }

                    queue(
                        &mut pending,
                        &r1.args,
                        &r2.args,
                        r1.fun == r2.fun && r1.forces == r2.forces && i1 == i2,
                    );
                }
                (
                    Value::Constr {
                        tag: t1,
                        fields: f1,
                        term_id: i1,
                    },
                    Value::Constr {
                        tag: t2,
                        fields: f2,
                        term_id: i2,
                    },
                ) => {
                    if t1 != t2 || f1.len() != f2.len() {
                        return false;
                    }

                    queue(&mut pending, f1, f2, i1 == i2);
                }
                _ => return false,
            }
        }

        true
    }
}

impl Value {
    /// A stand-in swapped into a machine frame when the real value is taken out of it.
    ///
    /// `Value` carries a manual `Drop` (see above), so the machine cannot move a value out of a
    /// frame by pattern any more. This leaf allocates nothing and drops for free.
    pub(crate) fn husk() -> Self {
        Value::Constr {
            tag: 0,
            fields: Vec::new(),
            term_id: 0,
        }
    }

    /// The source term id this value originated from, if any. A bare `Con` constant result carries
    /// no term id.
    pub fn term_id(&self) -> Option<isize> {
        match self {
            Value::Con(_) => None,
            Value::Delay { term_id, .. }
            | Value::Lambda { term_id, .. }
            | Value::Builtin { term_id, .. }
            | Value::Constr { term_id, .. } => Some(*term_id),
        }
    }

    pub fn integer(n: BigInt) -> Self {
        let constant = Constant::Integer(n);

        Value::Con(constant.into())
    }

    pub fn bool(n: bool) -> Self {
        let constant = Constant::Bool(n);

        Value::Con(constant.into())
    }

    pub fn byte_string(n: Vec<u8>) -> Self {
        let constant = Constant::ByteString(n);

        Value::Con(constant.into())
    }

    pub fn string(n: String) -> Self {
        let constant = Constant::String(n);

        Value::Con(constant.into())
    }

    pub fn list(typ: Type, n: Vec<Rc<Constant>>) -> Self {
        let constant = Constant::ProtoList(typ, n);

        Value::Con(constant.into())
    }

    pub fn array(typ: Type, n: Vec<Rc<Constant>>) -> Self {
        let constant = Constant::ProtoArray(typ, n);

        Value::Con(constant.into())
    }

    pub fn data(d: PlutusData) -> Self {
        let constant = Constant::Data(d);

        Value::Con(constant.into())
    }
    pub fn from_value(value: crate::ast::Value) -> Self {
        Value::Con(Constant::Value(value).into())
    }

    pub(super) fn unwrap_integer(&self) -> Result<&BigInt, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Integer(integer) = inner else {
            return Err(Error::TypeMismatch(Type::Integer, inner.into()));
        };

        Ok(integer)
    }

    pub(super) fn unwrap_byte_string(&self) -> Result<&Vec<u8>, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ByteString(byte_string) = inner else {
            return Err(Error::TypeMismatch(Type::ByteString, inner.into()));
        };

        Ok(byte_string)
    }

    pub(super) fn unwrap_string(&self) -> Result<&String, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::String(string) = inner else {
            return Err(Error::TypeMismatch(Type::String, inner.into()));
        };

        Ok(string)
    }

    pub(super) fn unwrap_bool(&self) -> Result<&bool, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bool(condition) = inner else {
            return Err(Error::TypeMismatch(Type::Bool, inner.into()));
        };

        Ok(condition)
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn unwrap_pair(
        &self,
    ) -> Result<(&Type, &Type, &Rc<Constant>, &Rc<Constant>), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoPair(t1, t2, first, second) = inner else {
            return Err(Error::PairTypeMismatch(inner.into()));
        };

        Ok((t1, t2, first, second))
    }

    pub(super) fn unwrap_list(&self) -> Result<(&Type, &Vec<Rc<Constant>>), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoList(t, list) = inner else {
            return Err(Error::ListTypeMismatch(inner.into()));
        };

        Ok((t, list))
    }

    pub(super) fn unwrap_array(&self) -> Result<(&Type, &Vec<Rc<Constant>>), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoArray(t, array) = inner else {
            return Err(Error::ArrayTypeMismatch(inner.into()));
        };

        Ok((t, array))
    }

    pub(super) fn unwrap_data(&self) -> Result<&PlutusData, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Data(data) = inner else {
            return Err(Error::TypeMismatch(Type::Data, inner.into()));
        };

        Ok(data)
    }
    pub(super) fn unwrap_value(&self) -> Result<&crate::ast::Value, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Value(value) = inner else {
            return Err(Error::TypeMismatch(Type::Value, inner.into()));
        };

        Ok(value)
    }

    pub(super) fn unwrap_unit(&self) -> Result<(), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Unit = inner else {
            return Err(Error::TypeMismatch(Type::Unit, inner.into()));
        };

        Ok(())
    }

    /// Like `unwrap_constant`, but hands out the shared allocation itself so
    /// the caller can store it (e.g. as a list element) without a deep clone.
    pub(super) fn unwrap_constant_rc(&self) -> Result<Rc<Constant>, Error> {
        let Value::Con(item) = self else {
            return Err(Error::NotAConstant(self.clone()));
        };

        Ok(Rc::clone(item))
    }

    pub(super) fn unwrap_constant(&self) -> Result<&Constant, Error> {
        let Value::Con(item) = self else {
            return Err(Error::NotAConstant(self.clone()));
        };

        Ok(item.as_ref())
    }

    pub(super) fn unwrap_data_list(&self) -> Result<&Vec<Rc<Constant>>, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoList(Type::Data, list) = inner else {
            return Err(Error::TypeMismatch(
                Type::List(Type::Data.into()),
                inner.into(),
            ));
        };

        Ok(list)
    }

    pub(super) fn unwrap_int_list(&self) -> Result<&Vec<Rc<Constant>>, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoList(Type::Integer, list) = inner else {
            return Err(Error::TypeMismatch(
                Type::List(Type::Integer.into()),
                inner.into(),
            ));
        };

        Ok(list)
    }

    pub(super) fn unwrap_bls12_381_g1_element(&self) -> Result<&blst::blst_p1, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bls12_381G1Element(element) = inner else {
            return Err(Error::TypeMismatch(Type::Bls12_381G1Element, inner.into()));
        };

        Ok(element)
    }

    pub(super) fn unwrap_bls12_381_g2_element(&self) -> Result<&blst::blst_p2, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bls12_381G2Element(element) = inner else {
            return Err(Error::TypeMismatch(Type::Bls12_381G2Element, inner.into()));
        };

        Ok(element)
    }

    pub(super) fn unwrap_bls12_381_ml_result(&self) -> Result<&blst::blst_fp12, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bls12_381MlResult(element) = inner else {
            return Err(Error::TypeMismatch(Type::Bls12_381MlResult, inner.into()));
        };

        Ok(element)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Value::Con(i) if matches!(i.as_ref(), Constant::Integer(_)))
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Con(b) if matches!(b.as_ref(), Constant::Bool(_)))
    }

    pub fn cost_as_size(&self, func: DefaultFunction) -> Result<i64, Error> {
        let size = self.unwrap_integer()?;

        if size.is_negative() {
            let error = match func {
                DefaultFunction::IntegerToByteString => {
                    Error::IntegerToByteStringNegativeSize(size.clone())
                }
                DefaultFunction::ReplicateByte => Error::ReplicateByteNegativeSize(size.clone()),
                _ => unreachable!(),
            };
            return Err(error);
        }

        if size > &BigInt::from(runtime::INTEGER_TO_BYTE_STRING_MAXIMUM_OUTPUT_LENGTH) {
            let error = match func {
                DefaultFunction::IntegerToByteString => Error::IntegerToByteStringSizeTooBig(
                    size.clone(),
                    runtime::INTEGER_TO_BYTE_STRING_MAXIMUM_OUTPUT_LENGTH,
                ),
                DefaultFunction::ReplicateByte => Error::ReplicateByteSizeTooBig(
                    size.clone(),
                    runtime::INTEGER_TO_BYTE_STRING_MAXIMUM_OUTPUT_LENGTH,
                ),
                _ => unreachable!(),
            };
            return Err(error);
        }

        let arg1: i64 = u64::try_from(size).unwrap().try_into().unwrap();

        let arg1_exmem = if arg1 == 0 { 0 } else { ((arg1 - 1) / 8) + 1 };

        Ok(arg1_exmem)
    }

    pub fn to_ex_mem(&self) -> i64 {
        self.to_ex_mem_with_semantics(BuiltinSemantics::C)
    }

    pub fn to_ex_mem_with_semantics(&self, semantics: BuiltinSemantics) -> i64 {
        match self {
            Value::Con(c) => Self::constant_to_ex_mem(c, semantics),
            Value::Delay { .. } => 1,
            Value::Lambda { .. } => 1,
            Value::Builtin { .. } => 1,
            Value::Constr { .. } => 1,
        }
    }

    fn constant_to_ex_mem(constant: &Constant, semantics: BuiltinSemantics) -> i64 {
        let mut stack = vec![constant];
        let mut total = 0;

        while let Some(constant) = stack.pop() {
            match constant {
                Constant::Integer(i) => total += Self::integer_to_ex_mem(i),
                Constant::ByteString(b) => total += Self::byte_string_to_ex_mem(b),
                Constant::String(s) => {
                    total += if semantics.costs_strings_by_utf8_bytes() {
                        Self::utf8_text_to_ex_mem(s)
                    } else {
                        s.chars().count() as i64
                    };
                }
                Constant::Unit | Constant::Bool(_) => total += 1,
                Constant::ProtoList(_, items) => {
                    stack.extend(items.iter().map(|item| item.as_ref()))
                }
                // An array is measured by its length, not by the size of its elements.
                Constant::ProtoArray(_, items) => total += items.len() as i64,
                Constant::ProtoPair(_, _, l, r) => {
                    stack.push(l.as_ref());
                    stack.push(r.as_ref());
                }
                Constant::Data(item) => total += Self::data_to_ex_mem_inner(item),
                Constant::Bls12_381G1Element(_) => total += size_of::<blst::blst_p1>() as i64 / 8,
                Constant::Bls12_381G2Element(_) => total += size_of::<blst::blst_p2>() as i64 / 8,
                Constant::Bls12_381MlResult(_) => total += size_of::<blst::blst_fp12>() as i64 / 8,
                Constant::Value(value) => total += value.total_size() as i64,
            }
        }

        total
    }

    fn utf8_text_to_ex_mem(s: &str) -> i64 {
        (s.len() as i64) / 4
    }

    fn integer_to_ex_mem(i: &BigInt) -> i64 {
        if i.is_zero() {
            1
        } else {
            (integer_log2(i.abs()) / 64) + 1
        }
    }

    fn byte_string_to_ex_mem(b: &[u8]) -> i64 {
        if b.is_empty() {
            1
        } else {
            ((b.len() as i64 - 1) / 8) + 1
        }
    }

    pub fn data_to_ex_mem(&self, data: &PlutusData) -> i64 {
        Self::data_to_ex_mem_inner(data)
    }

    fn data_to_ex_mem_inner(data: &PlutusData) -> i64 {
        let mut stack: VecDeque<&PlutusData> = VecDeque::new();
        let mut total = 0;
        stack.push_front(data);

        while let Some(item) = stack.pop_front() {
            // each time we deconstruct a data we add 4 memory units
            total += 4;
            match item {
                PlutusData::Constr(c) => {
                    // note currently tag is not factored into cost of memory
                    // create new stack with of items from the list of data
                    let mut new_stack: VecDeque<&PlutusData> =
                        VecDeque::from_iter(c.fields.deref().iter());
                    // Append old stack to the back of the new stack
                    new_stack.append(&mut stack);
                    stack = new_stack;
                }
                PlutusData::Map(m) => {
                    let mut new_stack: VecDeque<&PlutusData>;
                    // create new stack with of items from the list of pairs of data
                    new_stack = m.iter().fold(VecDeque::new(), |mut acc, d| {
                        acc.push_back(&d.0);
                        acc.push_back(&d.1);
                        acc
                    });
                    // Append old stack to the back of the new stack
                    new_stack.append(&mut stack);
                    stack = new_stack;
                }
                PlutusData::BigInt(i) => {
                    let i = from_pallas_bigint(i);

                    total += Self::integer_to_ex_mem(&i);
                }
                PlutusData::BoundedBytes(b) => {
                    total += Self::byte_string_to_ex_mem(b.deref());
                }
                PlutusData::Array(a) => {
                    // create new stack with of items from the list of data
                    let mut new_stack: VecDeque<&PlutusData> =
                        VecDeque::from_iter(a.deref().iter());
                    // Append old stack to the back of the new stack
                    new_stack.append(&mut stack);
                    stack = new_stack;
                }
            }
        }
        total
    }
    pub(super) fn data_node_count(&self) -> Result<i64, Error> {
        let data = self.unwrap_data()?;
        let mut stack = VecDeque::from([data]);
        let mut count = 0_i64;

        while let Some(item) = stack.pop_front() {
            count += 1;
            match item {
                PlutusData::Constr(constr) => stack.extend(constr.fields.iter()),
                PlutusData::Map(entries) => {
                    for (key, value) in entries.iter() {
                        stack.push_back(key);
                        stack.push_back(value);
                    }
                }
                PlutusData::Array(items) => stack.extend(items.iter()),
                PlutusData::BigInt(_) | PlutusData::BoundedBytes(_) => {}
            }
        }

        Ok(count)
    }

    pub fn expect_type(&self, r#type: Type) -> Result<(), Error> {
        let constant: Constant = self.clone().try_into()?;

        let constant_type = Type::from(&constant);

        if constant_type == r#type {
            Ok(())
        } else {
            Err(Error::TypeMismatch(r#type, constant_type))
        }
    }

    pub fn expect_list(&self) -> Result<(), Error> {
        let constant: Constant = self.clone().try_into()?;

        let constant_type = Type::from(&constant);

        if matches!(constant_type, Type::List(_)) {
            Ok(())
        } else {
            Err(Error::ListTypeMismatch(constant_type))
        }
    }

    pub fn expect_pair(&self) -> Result<(), Error> {
        let constant: Constant = self.clone().try_into()?;

        let constant_type = Type::from(&constant);

        if matches!(constant_type, Type::Pair(_, _)) {
            Ok(())
        } else {
            Err(Error::PairTypeMismatch(constant_type))
        }
    }
}

impl TryFrom<Value> for Type {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let constant: Constant = value.try_into()?;

        let constant_type = Type::from(&constant);

        Ok(constant_type)
    }
}

impl TryFrom<&Value> for Type {
    type Error = Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let constant: Constant = value.try_into()?;

        let constant_type = Type::from(&constant);

        Ok(constant_type)
    }
}

impl TryFrom<Value> for Constant {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        // `Value` has a manual `Drop` (see above), so its fields cannot be moved out by pattern;
        // the constant is cloned out of the borrow, exactly as before.
        match &value {
            Value::Con(constant) => Ok(constant.as_ref().clone()),
            _ => Err(Error::NotAConstant(value)),
        }
    }
}

impl TryFrom<&Value> for Constant {
    type Error = Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::Con(constant) => Ok(constant.as_ref().clone()),
            rest => Err(Error::NotAConstant(rest.clone())),
        }
    }
}

pub fn integer_log2(i: BigInt) -> i64 {
    if i.is_zero() {
        return 0;
    }

    let (_, bytes) = i.to_bytes_be();

    match bytes.first() {
        None => unreachable!("empty number?"),
        Some(u) => (8 - u.leading_zeros() - 1) as i64 + 8 * (bytes.len() - 1) as i64,
    }
}

pub fn from_pallas_bigint(n: &conway::BigInt) -> BigInt {
    match n {
        conway::BigInt::Int(i) => i128::from(*i).into(),
        conway::BigInt::BigUInt(bytes) => BigInt::from_bytes_be(num_bigint::Sign::Plus, bytes),
        conway::BigInt::BigNInt(bytes) => BigInt::from_bytes_be(num_bigint::Sign::Minus, bytes) - 1,
    }
}

pub fn to_pallas_bigint(n: &BigInt) -> conway::BigInt {
    if let Some(i) = n.to_i128()
        && let Ok(i) = i.try_into()
    {
        let pallas_int: pallas_codec::utils::Int = i;
        return conway::BigInt::Int(pallas_int);
    }

    if n.is_positive() {
        let (_, bytes) = n.to_bytes_be();
        conway::BigInt::BigUInt(bytes.into())
    } else {
        // Note that this would break if n == 0
        // BUT n == 0 always fits into 64bits and hence would end up in the first branch.
        let n: BigInt = n + 1;
        let (_, bytes) = n.to_bytes_be();
        conway::BigInt::BigNInt(bytes.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, NamedDeBruijn, Term, Type},
        machine::{
            runtime::{BuiltinRuntime, BuiltinSemantics},
            value::{Env, Value, integer_log2},
        },
    };
    use num_bigint::BigInt;
    use std::rc::Rc;

    /// Deep enough that a destructor taking one stack frame per level cannot survive the test
    /// harness's ~2 MiB thread stack, let alone the ~1 MiB a `wasm32` engine gives us.
    const DEEP: usize = 100_000;

    #[test]
    fn to_ex_mem_bigint() {
        let value = Value::Con(Constant::Integer(1.into()).into());

        assert_eq!(value.to_ex_mem(), 1);

        let value = Value::Con(Constant::Integer(42.into()).into());

        assert_eq!(value.to_ex_mem(), 1);

        let value = Value::Con(
            Constant::Integer(BigInt::parse_bytes("18446744073709551615".as_bytes(), 10).unwrap())
                .into(),
        );

        assert_eq!(value.to_ex_mem(), 1);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("999999999999999999999999999999".as_bytes(), 10).unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105726".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105727".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105728".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105729".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("340282366920938463463374607431768211458".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 3);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("999999999999999999999999999999999999999999".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 3);

        let value =
            Value::Con(Constant::Integer(BigInt::parse_bytes("999999999999999999999999999999999999999999999999999999999999999999999999999999999999".as_bytes(), 10).unwrap()).into());

        assert_eq!(value.to_ex_mem(), 5);
    }

    #[test]
    fn integer_log2_oracle() {
        // Values come from the Haskell implementation
        assert_eq!(integer_log2(0.into()), 0);
        assert_eq!(integer_log2(1.into()), 0);
        assert_eq!(integer_log2(42.into()), 5);
        assert_eq!(
            integer_log2(BigInt::parse_bytes("18446744073709551615".as_bytes(), 10).unwrap()),
            63
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("999999999999999999999999999999".as_bytes(), 10).unwrap()
            ),
            99
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("170141183460469231731687303715884105726".as_bytes(), 10)
                    .unwrap()
            ),
            126
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("170141183460469231731687303715884105727".as_bytes(), 10)
                    .unwrap()
            ),
            126
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("170141183460469231731687303715884105728".as_bytes(), 10)
                    .unwrap()
            ),
            127
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("340282366920938463463374607431768211458".as_bytes(), 10)
                    .unwrap()
            ),
            128
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("999999999999999999999999999999999999999999".as_bytes(), 10)
                    .unwrap()
            ),
            139
        );
        assert_eq!(
            integer_log2(BigInt::parse_bytes("999999999999999999999999999999999999999999999999999999999999999999999999999999999999".as_bytes(), 10).unwrap()),
            279
        );
    }

    #[test]
    fn to_ex_mem_counts_nested_constants_iteratively() {
        let nested = Constant::ProtoPair(
            Type::Integer,
            Type::List(Type::String.into()),
            Rc::new(Constant::Integer((1_i128 << 64).into())),
            Rc::new(Constant::ProtoList(
                Type::String,
                vec![
                    Constant::String("abcd".to_string()).into(),
                    Constant::String("é".to_string()).into(),
                    Constant::ByteString(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).into(),
                ],
            )),
        );

        let value = Value::Con(nested.into());

        assert_eq!(value.to_ex_mem_with_semantics(BuiltinSemantics::C), 9);
        assert_eq!(value.to_ex_mem_with_semantics(BuiltinSemantics::D), 5);
    }

    /// Both spines below are built with a loop, never a recursive helper: a recursive builder
    /// would overflow while setting the test up.
    #[test]
    fn deep_environment_chain_drops_without_overflowing_the_stack() {
        // Every lambda the machine enters extends the environment, and each captured environment
        // holds values that capture environments of their own.
        let mut env: Env = Rc::new(vec![]);

        for level in 0..DEEP {
            let value = Value::Delay {
                body: Rc::new(Term::<NamedDeBruijn>::Error { uniq_id: 0 }),
                env,
                term_id: level as isize,
            };

            env = Rc::new(vec![value]);
        }

        drop(env);
    }

    #[test]
    fn deep_constr_value_drops_without_overflowing_the_stack() {
        let mut value = Value::Con(Rc::new(Constant::Unit));

        for level in 0..DEEP {
            value = Value::Constr {
                tag: 0,
                fields: vec![value],
                term_id: level as isize,
            };
        }

        drop(value);
    }

    #[test]
    fn deep_values_clone_and_compare_without_overflowing_the_stack() {
        // Fields, builtin arguments and environments, alternating, so every way a value nests is
        // followed.
        let build = |leaf: Value| {
            let mut value = leaf;

            for level in 0..DEEP {
                value = match level % 3 {
                    0 => Value::Constr {
                        tag: 0,
                        fields: vec![value],
                        term_id: level as isize,
                    },
                    1 => Value::Builtin {
                        fun: crate::builtins::DefaultFunction::IfThenElse,
                        runtime: BuiltinRuntime {
                            args: vec![value],
                            fun: crate::builtins::DefaultFunction::IfThenElse,
                            forces: 1,
                        },
                        term_id: level as isize,
                    },
                    _ => Value::Lambda {
                        parameter_name: NamedDeBruijn {
                            text: "x".to_string(),
                            index: 0.into(),
                        }
                        .into(),
                        body: Rc::new(Term::<NamedDeBruijn>::Error { uniq_id: 0 }),
                        env: Rc::new(vec![value]),
                        term_id: level as isize,
                    },
                };
            }

            value
        };

        let value = build(Value::Con(Rc::new(Constant::Integer(1.into()))));

        let copy = value.clone();
        assert!(copy == value);
        assert!(value != build(Value::Con(Rc::new(Constant::Integer(2.into())))));
    }

    #[test]
    fn value_clone_shares_rc_children() {
        let env: Env = Rc::new(vec![Value::Con(Rc::new(Constant::Unit))]);
        let value = Value::Constr {
            tag: 1,
            fields: vec![Value::Delay {
                body: Rc::new(Term::<NamedDeBruijn>::Error { uniq_id: 3 }),
                env: Rc::clone(&env),
                term_id: 4,
            }],
            term_id: 5,
        };

        let copy = value.clone();

        assert!(copy == value);
        assert_eq!(Rc::strong_count(&env), 3);
        assert_eq!(copy.term_id(), Some(5));
    }

    #[test]
    fn dropping_a_value_leaves_a_shared_environment_alone() {
        let shared: Env = Rc::new(vec![Value::Con(Rc::new(Constant::Bool(true)))]);

        let first = Value::Delay {
            body: Rc::new(Term::<NamedDeBruijn>::Error { uniq_id: 0 }),
            env: Rc::clone(&shared),
            term_id: 1,
        };

        let second = Value::Delay {
            body: Rc::new(Term::<NamedDeBruijn>::Error { uniq_id: 0 }),
            env: Rc::clone(&shared),
            term_id: 2,
        };

        assert_eq!(Rc::strong_count(&shared), 3);

        drop(first);

        // The environment `second` still reads was not emptied out from under it.
        assert_eq!(Rc::strong_count(&shared), 2);
        assert_eq!(shared.len(), 1);

        drop(second);

        assert_eq!(Rc::strong_count(&shared), 1);
        assert_eq!(shared.len(), 1);
    }
}
