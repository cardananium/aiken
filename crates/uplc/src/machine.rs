use std::{fmt::Display, mem, rc::Rc};

use crate::ast::{Constant, NamedDeBruijn, Term, Type};

pub mod cost_model;
pub mod discharge;
mod error;
pub mod eval_result;
pub mod runtime;
pub mod value;

use cost_model::{ExBudget, StepKind};
pub use error::Error;
use pallas_primitives::conway::Language;

use self::{
    cost_model::CostModel,
    runtime::{BuiltinRuntime, BuiltinSemantics},
    value::{Env, Value},
};

#[derive(Clone, Debug)]
pub enum MachineState {
    Return(Context, Value),
    Compute(Context, Env, Term<NamedDeBruijn>),
    Done(Term<NamedDeBruijn>),
}

#[derive(Clone, Debug)]
pub enum Context {
    FrameAwaitArg(Value, Box<Context>),
    FrameAwaitFunTerm(Env, Term<NamedDeBruijn>, Box<Context>),
    FrameAwaitFunValue(Value, Box<Context>),
    FrameForce(Box<Context>),
    FrameConstr(
        Env,
        usize,
        Vec<Term<NamedDeBruijn>>,
        Vec<Value>,
        Box<Context>,
        isize,
    ),
    FrameCases(Env, Vec<Term<NamedDeBruijn>>, Box<Context>),
    NoFrame,
}

/// The machine's control stack is a linked list of frames, one per pending application, force or
/// case — so its length tracks the nesting depth of the term being evaluated, which is
/// attacker-controlled just as the term itself is (see the `Drop` impl on `Term` in `ast.rs` for
/// the full rationale). The derived destructor would unwind it one stack frame per machine frame,
/// which is not survivable on `wasm32`. Do not remove this as redundant.
impl Drop for Context {
    fn drop(&mut self) {
        let mut pending: Vec<Context> = Vec::new();

        detach_context_child(self, &mut pending);

        while let Some(mut context) = pending.pop() {
            detach_context_child(&mut context, &mut pending);
        }
    }
}

/// Move the frame below `context` onto `pending`, leaving `context` at the bottom of the stack.
///
/// Unlike the `Rc` children of a term, a frame's tail is a `Box` and is therefore owned outright:
/// there is never another holder to consider, so it is always ours to take. The `NoFrame` left
/// behind is what makes the shallow `Drop` of the husk terminate immediately.
fn detach_context_child(context: &mut Context, pending: &mut Vec<Context>) {
    let tail = match context {
        Context::NoFrame => return,
        Context::FrameAwaitArg(_, tail)
        | Context::FrameAwaitFunTerm(_, _, tail)
        | Context::FrameAwaitFunValue(_, tail)
        | Context::FrameForce(tail)
        | Context::FrameConstr(_, _, _, _, tail, _)
        | Context::FrameCases(_, _, tail) => tail,
    };

    let tail = take_context_tail(tail);

    if !matches!(tail, Context::NoFrame) {
        pending.push(tail);
    }
}

/// Unlink a frame's tail, leaving `NoFrame` behind.
///
/// `Context` carries a manual `Drop` (see above), so the machine cannot move a tail out of a frame
/// by pattern any more. Taking it this way is the same cost — no allocation, and the husk left
/// behind drops for free.
pub(crate) fn take_context_tail(tail: &mut Box<Context>) -> Context {
    mem::replace(tail.as_mut(), Context::NoFrame)
}

pub const TERM_COUNT: usize = 9;
pub const BUILTIN_COUNT: usize = 87;

#[derive(Debug, Clone)]
pub enum Trace {
    Log(String),
    Label(String),
}

impl Display for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trace::Log(log) => f.write_str(log),
            Trace::Label(label) => f.write_str(label),
        }
    }
}

impl Trace {
    pub fn unwrap_log(self) -> Option<String> {
        match self {
            Trace::Log(log) => Some(log),
            _ => None,
        }
    }

    pub fn unwrap_label(self) -> Option<String> {
        match self {
            Trace::Label(label) => Some(label),
            _ => None,
        }
    }
}

pub struct Machine {
    costs: CostModel,
    pub ex_budget: ExBudget,
    slippage: u32,
    unbudgeted_steps: [u32; 10],
    pub traces: Vec<Trace>,
    pub spend_counter: Option<[i64; (TERM_COUNT + BUILTIN_COUNT) * 2]>,
    semantics: BuiltinSemantics,
}

impl Machine {
    pub fn new(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        let semantics = BuiltinSemantics::for_language(&version);

        Machine {
            costs,
            ex_budget: initial_budget,
            slippage,
            unbudgeted_steps: [0; 10],
            traces: vec![],
            spend_counter: None,
            semantics,
        }
    }

    pub fn new_with_protocol(
        version: Language,
        protocol_major_version: u16,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        let semantics =
            BuiltinSemantics::for_language_and_protocol(&version, protocol_major_version);

        Machine {
            costs,
            ex_budget: initial_budget,
            slippage,
            unbudgeted_steps: [0; 10],
            traces: vec![],
            spend_counter: None,
            semantics,
        }
    }

    pub fn new_debug(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        let semantics = BuiltinSemantics::for_language(&version);

        Machine {
            costs,
            ex_budget: initial_budget,
            slippage,
            unbudgeted_steps: [0; 10],
            traces: vec![],
            spend_counter: Some([0; (TERM_COUNT + BUILTIN_COUNT) * 2]),
            semantics,
        }
    }

    pub fn new_debug_with_protocol(
        version: Language,
        protocol_major_version: u16,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        let semantics =
            BuiltinSemantics::for_language_and_protocol(&version, protocol_major_version);

        Machine {
            costs,
            ex_budget: initial_budget,
            slippage,
            unbudgeted_steps: [0; 10],
            traces: vec![],
            spend_counter: Some([0; (TERM_COUNT + BUILTIN_COUNT) * 2]),
            semantics,
        }
    }

    pub fn run(&mut self, term: Term<NamedDeBruijn>) -> Result<Term<NamedDeBruijn>, Error> {
        use MachineState::*;

        let startup_budget = self.costs.machine_costs.get(StepKind::StartUp);

        self.spend_budget(startup_budget)?;

        let mut state = Compute(Context::NoFrame, Rc::new(vec![]), term);

        loop {
            state = match state {
                Compute(context, env, t) => self.compute(context, env, t)?,
                Return(context, value) => self.return_compute(context, value)?,
                Done(t) => {
                    return Ok(t);
                }
            };
        }
    }

    fn compute(
        &mut self,
        context: Context,
        env: Env,
        mut term: Term<NamedDeBruijn>,
    ) -> Result<MachineState, Error> {
        // `Term` carries a manual, iterative `Drop` (see `ast.rs`), which makes the compiler
        // reject moving fields out of it by pattern. Matching on `&mut` instead costs nothing:
        // every field taken here is either `Copy`, a cheap `Rc` handle, or a vector we can take
        // outright.
        match &mut term {
            Term::Var { name, uniq_id } => {
                self.step_and_maybe_spend(StepKind::Var)?;

                let val = self.lookup_var(name.as_ref(), &env, *uniq_id)?;

                Ok(MachineState::Return(context, val))
            }
            Term::Delay { body, uniq_id } => {
                self.step_and_maybe_spend(StepKind::Delay)?;

                Ok(MachineState::Return(context, Value::Delay {
                    body: Rc::clone(body),
                    env,
                    term_id: *uniq_id,
                }))
            }
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => {
                self.step_and_maybe_spend(StepKind::Lambda)?;

                Ok(MachineState::Return(
                    context,
                    Value::Lambda {
                        parameter_name: Rc::clone(parameter_name),
                        body: Rc::clone(body),
                        env,
                        term_id: *uniq_id,
                    },
                ))
            }
            Term::Apply { function, argument, .. } => {
                self.step_and_maybe_spend(StepKind::Apply)?;

                Ok(MachineState::Compute(
                    Context::FrameAwaitFunTerm(
                        env.clone(),
                        argument.as_ref().clone(),
                        context.into(),
                    ),
                    env,
                    function.as_ref().clone(),
                ))
            }
            Term::Constant { value, .. } => {
                self.step_and_maybe_spend(StepKind::Constant)?;

                Ok(MachineState::Return(context, Value::Con(Rc::clone(value))))
            }
            Term::Force { body, .. } => {
                self.step_and_maybe_spend(StepKind::Force)?;

                Ok(MachineState::Compute(
                    Context::FrameForce(context.into()),
                    env,
                    body.as_ref().clone(),
                ))
            }
            Term::Error { .. } => Err(Error::EvaluationFailure),
            Term::Builtin { fun, uniq_id } => {
                self.step_and_maybe_spend(StepKind::Builtin)?;

                let fun = *fun;

                let runtime: BuiltinRuntime = fun.into();

                Ok(MachineState::Return(
                    context,
                    Value::Builtin { fun, runtime, term_id: *uniq_id },
                ))
            }
            Term::Constr { tag, fields, uniq_id } => {
                self.step_and_maybe_spend(StepKind::Constr)?;

                let (tag, uniq_id) = (*tag, *uniq_id);
                let mut fields = mem::take(fields);

                fields.reverse();

                if !fields.is_empty() {
                    let popped_field = fields.pop().unwrap();

                    Ok(MachineState::Compute(
                        Context::FrameConstr(env.clone(), tag, fields, vec![], context.into(), uniq_id),
                        env,
                        popped_field,
                    ))
                } else {
                    Ok(MachineState::Return(
                        context,
                        Value::Constr {
                            tag,
                            fields: vec![],
                            term_id: uniq_id,
                        },
                    ))
                }
            }
            Term::Case { constr, branches, .. } => {
                self.step_and_maybe_spend(StepKind::Case)?;

                Ok(MachineState::Compute(
                    Context::FrameCases(env.clone(), mem::take(branches), context.into()),
                    env,
                    constr.as_ref().clone(),
                ))
            }
        }
    }

    fn return_compute(
        &mut self,
        mut context: Context,
        mut value: Value,
    ) -> Result<MachineState, Error> {
        // `Context` and `Value` both carry a manual, iterative `Drop` (see above and
        // `machine/value.rs`), which makes the compiler reject moving their fields out by pattern.
        // Every arm below takes only what it needs and leaves a free-to-drop husk in its place:
        // `NoFrame` for a tail, an empty `Constr` for a value, `Error` for a term.
        match &mut context {
            Context::NoFrame => {
                if self.unbudgeted_steps[9] > 0 {
                    self.spend_unbudgeted_steps()?;
                }

                let term = discharge::value_as_term(value);

                Ok(MachineState::Done(term))
            }
            Context::FrameForce(ctx) => {
                let ctx = take_context_tail(ctx);

                self.force_evaluate(ctx, value)
            }
            Context::FrameAwaitFunTerm(arg_env, arg, ctx) => {
                let arg_env = Rc::clone(arg_env);
                let arg = mem::replace(arg, Term::Error { uniq_id: 0 });
                // This is one of the two places the tail has to stay boxed, so it is re-boxed
                // rather than unlinked in place.
                let ctx = mem::replace(ctx, Box::new(Context::NoFrame));

                Ok(MachineState::Compute(
                    Context::FrameAwaitArg(value, ctx),
                    arg_env,
                    arg,
                ))
            }
            Context::FrameAwaitArg(fun, ctx) => {
                let fun = mem::replace(fun, Value::husk());
                let ctx = take_context_tail(ctx);

                self.apply_evaluate(ctx, fun, value)
            }
            Context::FrameAwaitFunValue(arg, ctx) => {
                let arg = mem::replace(arg, Value::husk());
                let ctx = take_context_tail(ctx);

                self.apply_evaluate(ctx, value, arg)
            }
            Context::FrameConstr(env, tag, fields, resolved_fields, ctx, term_id) => {
                let (env, tag, term_id) = (Rc::clone(env), *tag, *term_id);
                let mut fields = mem::take(fields);
                let mut resolved_fields = mem::take(resolved_fields);

                resolved_fields.push(value);

                if !fields.is_empty() {
                    let popped_field = fields.pop().unwrap();
                    let ctx = mem::replace(ctx, Box::new(Context::NoFrame));

                    Ok(MachineState::Compute(
                        Context::FrameConstr(env.clone(), tag, fields, resolved_fields, ctx, term_id),
                        env,
                        popped_field,
                    ))
                } else {
                    Ok(MachineState::Return(
                        take_context_tail(ctx),
                        Value::Constr {
                            tag,
                            fields: resolved_fields,
                            term_id,
                        },
                    ))
                }
            }
            Context::FrameCases(env, branches, ctx) => {
                let env = Rc::clone(env);
                let branches = mem::take(branches);
                let ctx = take_context_tail(ctx);

                // The scrutinee's fields are only taken on the branch that consumes them; the
                // error branches hand the value on untouched.
                match &mut value {
                    Value::Constr { tag, fields, .. } => match branches.get(*tag) {
                        Some(t) => {
                            let t = t.clone();

                            Ok(MachineState::Compute(
                                transfer_arg_stack(mem::take(fields), ctx),
                                env,
                                t,
                            ))
                        }
                        None => Err(Error::MissingCaseBranch(branches, value)),
                    },
                    _ => Err(Error::NonConstrScrutinized(value)),
                }
            }
        }
    }

    fn force_evaluate(&mut self, context: Context, mut value: Value) -> Result<MachineState, Error> {
        // `Value` carries a manual, iterative `Drop` (see `machine/value.rs`), which makes the
        // compiler reject moving fields out of it by pattern. The runtime is swapped out for a
        // fresh, empty one instead — same cost, and the husk left behind drops for free.
        match &mut value {
            Value::Delay { body, env, .. } => Ok(MachineState::Compute(
                context,
                Rc::clone(env),
                body.as_ref().clone(),
            )),
            Value::Builtin { fun, runtime, term_id } => {
                let (fun, term_id) = (*fun, *term_id);
                let mut runtime = mem::replace(runtime, BuiltinRuntime::new(fun));

                if runtime.needs_force() {
                    runtime.consume_force();

                    let res = if runtime.is_ready() {
                        self.eval_builtin_app(runtime)?
                    } else {
                        Value::Builtin { fun, runtime, term_id }
                    };

                    Ok(MachineState::Return(context, res))
                } else {
                    let term = discharge::value_as_term(Value::Builtin {
                        fun,
                        runtime,
                        term_id,
                    });

                    Err(Error::BuiltinTermArgumentExpected(term))
                }
            }
            _ => Err(Error::NonPolymorphicInstantiation(value)),
        }
    }

    fn apply_evaluate(
        &mut self,
        context: Context,
        mut function: Value,
        argument: Value,
    ) -> Result<MachineState, Error> {
        // `Value` carries a manual, iterative `Drop` (see `machine/value.rs`), which makes the
        // compiler reject moving fields out of it by pattern; the payloads are taken out
        // explicitly instead.
        match &mut function {
            Value::Lambda { body, env, .. } => {
                let body = Rc::clone(body);

                let e = Rc::make_mut(env);

                e.push(argument);

                Ok(MachineState::Compute(
                    context,
                    Rc::new(e.clone()),
                    body.as_ref().clone(),
                ))
            }
            Value::Builtin { fun, runtime, term_id } => {
                let (fun, term_id) = (*fun, *term_id);
                let runtime = mem::replace(runtime, BuiltinRuntime::new(fun));

                if runtime.is_arrow() && !runtime.needs_force() {
                    let mut runtime = runtime;

                    runtime.push(argument)?;

                    let res = if runtime.is_ready() {
                        self.eval_builtin_app(runtime)?
                    } else {
                        Value::Builtin { fun, runtime, term_id }
                    };

                    Ok(MachineState::Return(context, res))
                } else {
                    let term = discharge::value_as_term(Value::Builtin {
                        fun,
                        runtime,
                        term_id,
                    });

                    Err(Error::UnexpectedBuiltinTermArgument(term))
                }
            }
            _ => Err(Error::NonFunctionalApplication(function, argument)),
        }
    }

    fn eval_builtin_app(&mut self, runtime: BuiltinRuntime) -> Result<Value, Error> {
        let cost = runtime.to_ex_budget(&self.costs.builtin_costs, self.semantics)?;

        self.spend_budget(cost)?;

        if let Some(counter) = &mut self.spend_counter {
            let i = (runtime.fun as usize + TERM_COUNT) * 2;

            counter[i] += cost.mem;
            counter[i + 1] += cost.cpu;
        }

        runtime.call(self.semantics, &mut self.traces)
    }

    fn lookup_var(&mut self, name: &NamedDeBruijn, env: &[Value], term_id: isize) -> Result<Value, Error> {
        env.get::<usize>(env.len() - usize::from(name.index))
            .cloned()
            .ok_or_else(|| {
                Error::OpenTermEvaluated(Term::Var {
                    name: name.clone().into(),
                    uniq_id: term_id,
                })
            })
    }

    fn step_and_maybe_spend(&mut self, step: StepKind) -> Result<(), Error> {
        let index = step as u8;
        self.unbudgeted_steps[index as usize] += 1;
        self.unbudgeted_steps[9] += 1;

        if self.unbudgeted_steps[9] >= self.slippage {
            self.spend_unbudgeted_steps()?;
        }

        Ok(())
    }

    fn spend_unbudgeted_steps(&mut self) -> Result<(), Error> {
        for i in 0..self.unbudgeted_steps.len() - 1 {
            let mut unspent_step_budget =
                self.costs.machine_costs.get(StepKind::try_from(i as u8)?);

            unspent_step_budget.occurrences(self.unbudgeted_steps[i] as i64);

            self.spend_budget(unspent_step_budget)?;

            self.unbudgeted_steps[i] = 0;

            if let Some(counter) = &mut self.spend_counter {
                counter[i * 2] += unspent_step_budget.mem;
                counter[i * 2 + 1] += unspent_step_budget.cpu;
            }
        }

        self.unbudgeted_steps[9] = 0;

        Ok(())
    }

    fn spend_budget(&mut self, spend_budget: ExBudget) -> Result<(), Error> {
        self.ex_budget.mem -= spend_budget.mem;
        self.ex_budget.cpu -= spend_budget.cpu;

        if self.ex_budget.mem < 0 || self.ex_budget.cpu < 0 {
            Err(Error::OutOfExError(self.ex_budget))
        } else {
            Ok(())
        }
    }
}

fn transfer_arg_stack(mut args: Vec<Value>, ctx: Context) -> Context {
    if args.is_empty() {
        ctx
    } else {
        let popped_field = args.pop().unwrap();

        transfer_arg_stack(args, Context::FrameAwaitFunValue(popped_field, ctx.into()))
    }
}

impl From<&Constant> for Type {
    fn from(constant: &Constant) -> Self {
        match constant {
            Constant::Integer(_) => Type::Integer,
            Constant::ByteString(_) => Type::ByteString,
            Constant::String(_) => Type::String,
            Constant::Unit => Type::Unit,
            Constant::Bool(_) => Type::Bool,
            Constant::ProtoList(t, _) => Type::List(Rc::new(t.clone())),
            Constant::ProtoPair(t1, t2, _, _) => {
                Type::Pair(Rc::new(t1.clone()), Rc::new(t2.clone()))
            }
            Constant::Data(_) => Type::Data,
            Constant::Bls12_381G1Element(_) => Type::Bls12_381G1Element,
            Constant::Bls12_381G2Element(_) => Type::Bls12_381G2Element,
            Constant::Bls12_381MlResult(_) => Type::Bls12_381MlResult,
        }
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::{Context, cost_model::ExBudget, runtime::Compressable};
    use crate::{
        ast::{Constant, NamedDeBruijn, Program, Term},
        builtins::DefaultFunction,
    };

    /// Deep enough that a destructor taking one stack frame per level cannot survive the test
    /// harness's ~2 MiB thread stack, let alone the ~1 MiB a `wasm32` engine gives us.
    const DEEP: usize = 100_000;

    /// The control stack grows one frame per pending force/apply, so its depth follows the nesting
    /// depth of the term being evaluated, which is attacker-controlled. Built with a loop on
    /// purpose: a recursive builder would overflow before the drop ever ran.
    #[test]
    fn deep_context_drops_without_overflowing_the_stack() {
        let mut context = Context::NoFrame;

        for _ in 0..DEEP {
            context = Context::FrameForce(Box::new(context));
        }

        drop(context);
    }

    #[test]
    fn add_big_ints() {
        let program: Program<NamedDeBruijn> = Program {
            version: (0, 0, 0),
            term: Term::Apply {
                uniq_id: 0,
                function: Term::Apply {
                    uniq_id: 1,
                    function: Term::Builtin {
                        fun: DefaultFunction::AddInteger,
                        uniq_id: 2,
                    }
                    .into(),
                    argument: Term::Constant {
                        value: Constant::Integer(i128::MAX.into()).into(),
                        uniq_id: 3,
                    }
                    .into(),
                }
                .into(),
                argument: Term::Constant {
                    value: Constant::Integer(i128::MAX.into()).into(),
                    uniq_id: 4,
                }
                .into(),
            },
        };

        let eval_result = program.eval(ExBudget::default());

        let term = eval_result.result().unwrap();

        assert_eq!(
            term,
            Term::Constant {
                value: Constant::Integer(
                    Into::<BigInt>::into(i128::MAX) + Into::<BigInt>::into(i128::MAX)
                )
                .into(),
                uniq_id: 0,
            }
        );
    }

    #[test]
    fn divide_integer() {
        let make_program = |fun: DefaultFunction, n: i32, m: i32| Program::<NamedDeBruijn> {
            version: (0, 0, 0),
            term: Term::Apply {
                uniq_id: 0,
                function: Term::Apply {
                    uniq_id: 1,
                    function: Term::Builtin {
                        fun,
                        uniq_id: 2,
                    }
                    .into(),
                    argument: Term::Constant {
                        value: Constant::Integer(n.into()).into(),
                        uniq_id: 3,
                    }
                    .into(),
                }
                .into(),
                argument: Term::Constant {
                    value: Constant::Integer(m.into()).into(),
                    uniq_id: 4,
                }
                .into(),
            },
        };

        let test_data = vec![
            (DefaultFunction::DivideInteger, 8, 3, 2),
            (DefaultFunction::DivideInteger, 8, -3, -3),
            (DefaultFunction::DivideInteger, -8, 3, -3),
            (DefaultFunction::DivideInteger, -8, -3, 2),
            (DefaultFunction::QuotientInteger, 8, 3, 2),
            (DefaultFunction::QuotientInteger, 8, -3, -2),
            (DefaultFunction::QuotientInteger, -8, 3, -2),
            (DefaultFunction::QuotientInteger, -8, -3, 2),
            (DefaultFunction::RemainderInteger, 8, 3, 2),
            (DefaultFunction::RemainderInteger, 8, -3, 2),
            (DefaultFunction::RemainderInteger, -8, 3, -2),
            (DefaultFunction::RemainderInteger, -8, -3, -2),
            (DefaultFunction::ModInteger, 8, 3, 2),
            (DefaultFunction::ModInteger, 8, -3, -1),
            (DefaultFunction::ModInteger, -8, 3, 1),
            (DefaultFunction::ModInteger, -8, -3, -2),
        ];

        for (fun, n, m, result) in test_data {
            let eval_result = make_program(fun, n, m).eval(ExBudget::default());

            assert_eq!(
                eval_result.result().unwrap(),
                Term::Constant {
                    value: Constant::Integer(result.into()).into(),
                    uniq_id: 0,
                }
            );
        }
    }

    #[test]
    fn case_constr_case_0() {
        let make_program =
            |fun: DefaultFunction, tag: usize, n: i32, m: i32| Program::<NamedDeBruijn> {
                version: (0, 0, 0),
                term: Term::Case {
                    constr: Term::Constr {
                        tag,
                        fields: vec![
                            Term::Constant {
                                value: Constant::Integer(n.into()).into(),
                                uniq_id: 0,
                            },
                            Term::Constant {
                                value: Constant::Integer(m.into()).into(),
                                uniq_id: 1,
                            },
                        ],
                        uniq_id: 2,
                    }
                    .into(),
                    branches: vec![
                        Term::Builtin {
                            fun,
                            uniq_id: 3,
                        },
                        Term::subtract_integer(),
                    ],
                    uniq_id: 4,
                },
            };

        let test_data = vec![
            (DefaultFunction::AddInteger, 0, 8, 3, 11),
            (DefaultFunction::AddInteger, 1, 8, 3, 5),
        ];

        for (fun, tag, n, m, result) in test_data {
            let eval_result = make_program(fun, tag, n, m).eval(ExBudget::max());

            assert_eq!(
                eval_result.result().unwrap(),
                Term::Constant {
                    value: Constant::Integer(result.into()).into(),
                    uniq_id: 0,
                }
            );
        }
    }

    #[test]
    fn case_constr_case_1() {
        let make_program = |tag: usize| Program::<NamedDeBruijn> {
            version: (0, 0, 0),
            term: Term::Case {
                constr: Term::Constr {
                    tag,
                    fields: vec![],
                    uniq_id: 0,
                }
                .into(),
                branches: vec![
                    Term::integer(5.into()),
                    Term::integer(10.into()),
                    Term::integer(15.into()),
                ],
                uniq_id: 1,
            },
        };

        let test_data = vec![(0, 5), (1, 10), (2, 15)];

        for (tag, result) in test_data {
            let eval_result = make_program(tag).eval(ExBudget::max());

            assert_eq!(
                eval_result.result().unwrap(),
                Term::Constant {
                    value: Constant::Integer(result.into()).into(),
                    uniq_id: 0,
                }
            );
        }
    }

    #[test]
    fn bls_g1_add_associative() {
        let a = blst::blst_p1::uncompress(&[
            0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac,
            0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2,
            0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00,
            0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f,
        ])
        .unwrap();

        let b = blst::blst_p1::uncompress(&[
            0x95, 0x0d, 0xfd, 0x33, 0xda, 0x26, 0x82, 0x26, 0x0c, 0x76, 0x03, 0x8d, 0xfb, 0x8b,
            0xad, 0x6e, 0x84, 0xae, 0x9d, 0x59, 0x9a, 0x3c, 0x15, 0x18, 0x15, 0x94, 0x5a, 0xc1,
            0xe6, 0xef, 0x6b, 0x10, 0x27, 0xcd, 0x91, 0x7f, 0x39, 0x07, 0x47, 0x9d, 0x20, 0xd6,
            0x36, 0xce, 0x43, 0x7a, 0x41, 0xf5,
        ])
        .unwrap();

        let c = blst::blst_p1::uncompress(&[
            0xb9, 0x62, 0xfd, 0x0c, 0xc8, 0x10, 0x48, 0xe0, 0xcf, 0x75, 0x57, 0xbf, 0x3e, 0x4b,
            0x6e, 0xdc, 0x5a, 0xb4, 0xbf, 0xb3, 0xdc, 0x87, 0xf8, 0x3a, 0xf4, 0x28, 0xb6, 0x30,
            0x07, 0x27, 0xb1, 0x39, 0xc4, 0x04, 0xab, 0x15, 0x9b, 0xdf, 0x2e, 0xae, 0xa3, 0xf6,
            0x49, 0x90, 0x34, 0x21, 0x53, 0x7f,
        ])
        .unwrap();

        let term: Term<NamedDeBruijn> = Term::bls12_381_g1_equal()
            .apply(
                Term::bls12_381_g1_add().apply(Term::bls12_381_g1(a)).apply(
                    Term::bls12_381_g1_add()
                        .apply(Term::bls12_381_g1(b))
                        .apply(Term::bls12_381_g1(c)),
                ),
            )
            .apply(
                Term::bls12_381_g1_add()
                    .apply(
                        Term::bls12_381_g1_add()
                            .apply(Term::bls12_381_g1(a))
                            .apply(Term::bls12_381_g1(b)),
                    )
                    .apply(Term::bls12_381_g1(c)),
            );

        let program = Program {
            version: (1, 0, 0),
            term,
        };

        let eval_result = program.eval(Default::default());

        let final_term = eval_result.result().unwrap();

        assert_eq!(final_term, Term::bool(true))
    }

    #[test]
    fn bls_g2_add_associative() {
        let a = blst::blst_p1::uncompress(&[
            0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac,
            0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2,
            0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00,
            0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f,
        ])
        .unwrap();

        let b = blst::blst_p1::uncompress(&[
            0x95, 0x0d, 0xfd, 0x33, 0xda, 0x26, 0x82, 0x26, 0x0c, 0x76, 0x03, 0x8d, 0xfb, 0x8b,
            0xad, 0x6e, 0x84, 0xae, 0x9d, 0x59, 0x9a, 0x3c, 0x15, 0x18, 0x15, 0x94, 0x5a, 0xc1,
            0xe6, 0xef, 0x6b, 0x10, 0x27, 0xcd, 0x91, 0x7f, 0x39, 0x07, 0x47, 0x9d, 0x20, 0xd6,
            0x36, 0xce, 0x43, 0x7a, 0x41, 0xf5,
        ])
        .unwrap();

        let c = blst::blst_p1::uncompress(&[
            0xb9, 0x62, 0xfd, 0x0c, 0xc8, 0x10, 0x48, 0xe0, 0xcf, 0x75, 0x57, 0xbf, 0x3e, 0x4b,
            0x6e, 0xdc, 0x5a, 0xb4, 0xbf, 0xb3, 0xdc, 0x87, 0xf8, 0x3a, 0xf4, 0x28, 0xb6, 0x30,
            0x07, 0x27, 0xb1, 0x39, 0xc4, 0x04, 0xab, 0x15, 0x9b, 0xdf, 0x2e, 0xae, 0xa3, 0xf6,
            0x49, 0x90, 0x34, 0x21, 0x53, 0x7f,
        ])
        .unwrap();

        let term: Term<NamedDeBruijn> = Term::bls12_381_g1_equal()
            .apply(
                Term::bls12_381_g1_add().apply(Term::bls12_381_g1(a)).apply(
                    Term::bls12_381_g1_add()
                        .apply(Term::bls12_381_g1(b))
                        .apply(Term::bls12_381_g1(c)),
                ),
            )
            .apply(
                Term::bls12_381_g1_add()
                    .apply(
                        Term::bls12_381_g1_add()
                            .apply(Term::bls12_381_g1(a))
                            .apply(Term::bls12_381_g1(b)),
                    )
                    .apply(Term::bls12_381_g1(c)),
            );

        let program = Program {
            version: (1, 0, 0),
            term,
        };

        let eval_result = program.eval(Default::default());

        let final_term = eval_result.result().unwrap();

        assert_eq!(final_term, Term::bool(true))
    }
}
