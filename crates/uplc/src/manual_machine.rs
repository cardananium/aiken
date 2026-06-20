use std::rc::Rc;

use crate::ast::{NamedDeBruijn, Term};
use crate::machine::{
    cost_model::{ExBudget, StepKind, CostModel},
    runtime::{BuiltinRuntime, BuiltinSemantics},
    value::{Value, Env},
    Context, MachineState, Trace, BUILTIN_COUNT, TERM_COUNT, Error,
};
use crate::machine::discharge::value_as_term;
use pallas_primitives::conway::Language;

/// Execution status of ManualMachine
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    /// Ready to execute the next step
    Ready,
    /// Execution completed with result
    Done(Term<NamedDeBruijn>),
    /// Execution stopped with error
    Error(Error),
}

#[derive(Clone, Debug)]
pub struct ManualMachine {
    /// Current machine state
    state: MachineState,
    /// Cost model
    costs: CostModel,
    /// Current execution budget
    pub ex_budget: ExBudget,
    /// Slippage parameter for grouping steps
    slippage: u32,
    /// Unbudgeted steps for each operation type
    unbudgeted_steps: [u32; 10],
    /// Execution traces
    pub traces: Vec<Trace>,
    /// Spend counter (optional for debugging)
    pub spend_counter: Option<[i64; (TERM_COUNT + BUILTIN_COUNT) * 2]>,
    /// Language version
    version: Language,
    /// Execution status
    status: ExecutionStatus,
}

impl ManualMachine {
    /// Creates a new ManualMachine
    pub fn new(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
        term: Term<NamedDeBruijn>,
    ) -> Result<Self, Error> {
        let startup_budget = costs.machine_costs.get(StepKind::StartUp);
        let mut ex_budget = initial_budget;
        
        // Deduct startup budget
        ex_budget.mem -= startup_budget.mem;
        ex_budget.cpu -= startup_budget.cpu;
        
        if ex_budget.mem < 0 || ex_budget.cpu < 0 {
            return Err(Error::OutOfExError(ex_budget));
        }

        let initial_state = MachineState::Compute(Context::NoFrame, Rc::new(vec![]), term);

        Ok(ManualMachine {
            state: initial_state,
            costs,
            ex_budget,
            slippage,
            unbudgeted_steps: [0; 10],
            traces: vec![],
            spend_counter: None,
            version,
            status: ExecutionStatus::Ready,
        })
    }

    /// Creates a new ManualMachine with debug information
    pub fn new_debug(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
        term: Term<NamedDeBruijn>,
    ) -> Result<Self, Error> {
        let mut machine = Self::new(version, costs, initial_budget, slippage, term)?;
        machine.spend_counter = Some([0; (TERM_COUNT + BUILTIN_COUNT) * 2]);
        Ok(machine)
    }

    /// Returns the current execution status
    pub fn status(&self) -> &ExecutionStatus {
        &self.status
    }

    /// Returns the current machine state
    pub fn current_state(&self) -> &MachineState {
        &self.state
    }

    /// Returns the current context (queue)
    pub fn current_context(&self) -> Option<&Context> {
        match &self.state {
            MachineState::Return(context, _) => Some(context),
            MachineState::Compute(context, _, _) => Some(context),
            MachineState::Done(_) => None,
        }
    }

    /// Returns the next term to execute (if any)
    pub fn next_term(&self) -> Option<&Term<NamedDeBruijn>> {
        match &self.state {
            MachineState::Compute(_, _, term) => Some(term),
            MachineState::Return(_, _) => None,
            MachineState::Done(term) => Some(term),
        }
    }

    /// Executes one step of computation
    pub fn step(&mut self) -> &ExecutionStatus {
        if !matches!(self.status, ExecutionStatus::Ready) {
            return &self.status;
        }

        // The replaced-in value is only a temporary placeholder so we can move `self.state` out
        // (mem::replace). It is overwritten on the Ready path and only survives into `self.state`
        // when a step errors — so it must be a STABLE sentinel, not `next_uniq_id()`. Using a fresh
        // global id here burned one id per step and, because the global counter never resets in a
        // long-lived host (e.g. the de-uplc-web worker), made the error term's id differ every run.
        // -1 matches the engine's "no specific term" sentinel and is fully deterministic.
        match std::mem::replace(&mut self.state, MachineState::Done(Term::Error { uniq_id: -1 })) {
            MachineState::Compute(context, env, term) => {
                // Capture the id of the term being computed before it is moved, so on failure the
                // resulting Error state names the actual source term that failed (e.g. a source
                // `(error)`) rather than the -1 sentinel.
                let failing_id = term.uniq_id();
                match self.compute(context, env, term) {
                    Ok(new_state) => {
                        self.state = new_state;
                        self.status = ExecutionStatus::Ready;
                    }
                    Err(error) => {
                        self.state = MachineState::Done(Term::Error { uniq_id: failing_id });
                        self.status = ExecutionStatus::Error(error);
                    }
                }
            }
            MachineState::Return(context, value) => {
                // The value being returned carries the id of the term it came from (a failing
                // builtin application, etc.); `Con` constants have none, so fall back to -1.
                let failing_id = value.term_id().unwrap_or(-1);
                match self.return_compute(context, value) {
                    Ok(new_state) => {
                        self.state = new_state;
                        self.status = ExecutionStatus::Ready;
                    }
                    Err(error) => {
                        self.state = MachineState::Done(Term::Error { uniq_id: failing_id });
                        self.status = ExecutionStatus::Error(error);
                    }
                }
            }
            MachineState::Done(term) => {
                self.state = MachineState::Done(term.clone());
                self.status = ExecutionStatus::Done(term);
            }
        }

        &self.status
    }

    /// Executes computation until completion or error
    pub fn run_to_completion(&mut self) -> &ExecutionStatus {
        loop {
            match self.step() {
                ExecutionStatus::Ready => continue,
                _ => break,
            }
        }
        &self.status
    }

    /// Executes a given number of steps
    pub fn step_n(&mut self, n: usize) -> &ExecutionStatus {
        for _ in 0..n {
            match self.step() {
                ExecutionStatus::Ready => continue,
                _ => break,
            }
        }
        &self.status
    }

    /// Resets the machine with a new term
    pub fn reset(&mut self, term: Term<NamedDeBruijn>) -> Result<(), Error> {
        let startup_budget = self.costs.machine_costs.get(StepKind::StartUp);
        
        // Check if we have enough budget for the new run
        if self.ex_budget.mem < startup_budget.mem || self.ex_budget.cpu < startup_budget.cpu {
            return Err(Error::OutOfExError(self.ex_budget));
        }

        // Deduct startup budget
        self.ex_budget.mem -= startup_budget.mem;
        self.ex_budget.cpu -= startup_budget.cpu;

        self.state = MachineState::Compute(Context::NoFrame, Rc::new(vec![]), term);
        self.status = ExecutionStatus::Ready;
        self.unbudgeted_steps = [0; 10];
        self.traces.clear();
        
        if let Some(counter) = &mut self.spend_counter {
            counter.fill(0);
        }

        Ok(())
    }

    /// Checks if the machine is ready to execute the next step
    pub fn is_ready(&self) -> bool {
        matches!(self.status, ExecutionStatus::Ready)
    }

    /// Checks if execution is completed
    pub fn is_done(&self) -> bool {
        matches!(self.status, ExecutionStatus::Done(_))
    }

    /// Checks if an error occurred
    pub fn is_error(&self) -> bool {
        matches!(self.status, ExecutionStatus::Error(_))
    }

    /// Collects all nested contexts from the current machine state as a list (iteratively)
    /// The contexts are returned in order from outermost to innermost
    /// 
    /// # Returns
    /// 
    /// A vector of contexts, where:
    /// - First element is the outermost context
    /// - Last element is the innermost context (usually NoFrame)
    /// - Empty vector if machine is in Done state
    /// 
    pub fn collect_nested_contexts(&self) -> Vec<Context> {
        let mut contexts = Vec::new();
        
        // Extract context from current state
        let mut current_context = match &self.state {
            MachineState::Return(context, _) => Some(context),
            MachineState::Compute(context, _, _) => Some(context), 
            MachineState::Done(_) => return contexts, // No context in Done state
        };
        
        // Iteratively traverse the context chain and collect all nested contexts
        while let Some(context) = current_context {
            contexts.push(context.clone());
            
            // Get the next nested context based on the variant
            current_context = match context {
                Context::NoFrame => None,
                Context::FrameForce(nested_ctx) => Some(nested_ctx.as_ref()),
                Context::FrameAwaitFunTerm(_, _, nested_ctx) => Some(nested_ctx.as_ref()),
                Context::FrameAwaitArg(_, nested_ctx) => Some(nested_ctx.as_ref()),
                Context::FrameAwaitFunValue(_, nested_ctx) => Some(nested_ctx.as_ref()),
                Context::FrameConstr(_, _, _, _, nested_ctx, _) => Some(nested_ctx.as_ref()),
                Context::FrameCases(_, _, nested_ctx) => Some(nested_ctx.as_ref()),
            };
        }
        
        contexts
    }

    // Methods copied from the original Machine
    fn compute(
        &mut self,
        context: Context,
        env: Env,
        term: Term<NamedDeBruijn>,
    ) -> Result<MachineState, Error> {
        match term {
            Term::Var { name, uniq_id } => {
                self.step_and_maybe_spend(StepKind::Var)?;

                let val = self.lookup_var(name.as_ref(), &env, uniq_id)?;

                Ok(MachineState::Return(context, val))
            }
            Term::Delay { body, uniq_id } => {
                self.step_and_maybe_spend(StepKind::Delay)?;

                Ok(MachineState::Return(context, Value::Delay { body, env, term_id: uniq_id }))
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
                        parameter_name,
                        body,
                        env,
                        term_id: uniq_id,
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

                Ok(MachineState::Return(context, Value::Con(value)))
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

                let runtime: BuiltinRuntime = fun.into();

                Ok(MachineState::Return(
                    context,
                    Value::Builtin { fun, runtime, term_id: uniq_id },
                ))
            }
            Term::Constr { tag, mut fields, uniq_id } => {
                self.step_and_maybe_spend(StepKind::Constr)?;

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
                    Context::FrameCases(env.clone(), branches, context.into()),
                    env,
                    constr.as_ref().clone(),
                ))
            }
        }
    }

    fn return_compute(&mut self, context: Context, value: Value) -> Result<MachineState, Error> {
        match context {
            Context::NoFrame => {
                if self.unbudgeted_steps[9] > 0 {
                    self.spend_unbudgeted_steps()?;
                }

                let term = value_as_term(value);

                Ok(MachineState::Done(term))
            }
            Context::FrameForce(ctx) => self.force_evaluate(*ctx, value),
            Context::FrameAwaitFunTerm(arg_env, arg, ctx) => Ok(MachineState::Compute(
                Context::FrameAwaitArg(value, ctx),
                arg_env,
                arg,
            )),
            Context::FrameAwaitArg(fun, ctx) => self.apply_evaluate(*ctx, fun, value),
            Context::FrameAwaitFunValue(arg, ctx) => self.apply_evaluate(*ctx, value, arg),
            Context::FrameConstr(env, tag, mut fields, mut resolved_fields, ctx, uniq_id) => {
                resolved_fields.push(value);

                if !fields.is_empty() {
                    let popped_field = fields.pop().unwrap();

                    Ok(MachineState::Compute(
                        Context::FrameConstr(env.clone(), tag, fields, resolved_fields, ctx, uniq_id),
                        env,
                        popped_field,
                    ))
                } else {
                    Ok(MachineState::Return(
                        *ctx,
                        Value::Constr {
                            tag,
                            fields: resolved_fields,
                            term_id: uniq_id,
                        },
                    ))
                }
            }
            Context::FrameCases(env, branches, ctx) => match value {
                Value::Constr { tag, fields, term_id } => match branches.get(tag) {
                    Some(t) => Ok(MachineState::Compute(
                        transfer_arg_stack(fields, *ctx),
                        env,
                        t.clone(),
                    )),
                    None => Err(Error::MissingCaseBranch(
                        branches,
                        Value::Constr { tag, fields, term_id },
                    )),
                },
                v => Err(Error::NonConstrScrutinized(v)),
            },
        }
    }

    fn force_evaluate(&mut self, context: Context, value: Value) -> Result<MachineState, Error> {
        match value {
            Value::Delay { body, env, .. } => {
                Ok(MachineState::Compute(context, env, body.as_ref().clone()))
            }
            Value::Builtin { fun, mut runtime, term_id } => {
                if runtime.needs_force() {
                    runtime.consume_force();

                    let res = if runtime.is_ready() {
                        self.eval_builtin_app(runtime)?
                    } else {
                        Value::Builtin { fun, runtime, term_id: term_id }
                    };

                    Ok(MachineState::Return(context, res))
                } else {
                    let term = value_as_term(Value::Builtin { fun, runtime, term_id: term_id });

                    Err(Error::BuiltinTermArgumentExpected(term))
                }
            }
            rest => Err(Error::NonPolymorphicInstantiation(rest)),
        }
    }

    fn apply_evaluate(
        &mut self,
        context: Context,
        function: Value,
        argument: Value,
    ) -> Result<MachineState, Error> {
        match function {
            Value::Lambda { body, mut env, .. } => {
                let e = Rc::make_mut(&mut env);

                e.push(argument);

                Ok(MachineState::Compute(
                    context,
                    Rc::new(e.clone()),
                    body.as_ref().clone(),
                ))
            }
            Value::Builtin { fun, runtime, term_id } => {
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
                    let term = value_as_term(Value::Builtin { fun, runtime, term_id });

                    Err(Error::UnexpectedBuiltinTermArgument(term))
                }
            }
            rest => Err(Error::NonFunctionalApplication(rest, argument)),
        }
    }

    fn eval_builtin_app(&mut self, runtime: BuiltinRuntime) -> Result<Value, Error> {
        let semantics = BuiltinSemantics::for_language(&self.version);

        let cost = runtime.to_ex_budget(&self.costs.builtin_costs, semantics)?;

        self.spend_budget(cost)?;

        if let Some(counter) = &mut self.spend_counter {
            let i = (runtime.fun as usize + TERM_COUNT) * 2;

            counter[i] += cost.mem;
            counter[i + 1] += cost.cpu;
        }

        runtime.call(semantics, &mut self.traces)
    }

    fn lookup_var(&mut self, name: &NamedDeBruijn, env: &[Value], term_id: isize) -> Result<Value, Error> {
        env.get::<usize>(env.len() - usize::from(name.index))
            .cloned()
            .ok_or_else(|| Error::OpenTermEvaluated(Term::Var { name: name.clone().into(), uniq_id: term_id }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Constant, Program};
    use crate::builtins::DefaultFunction;
    use crate::machine::cost_model::ExBudget;
    use num_bigint::BigInt;

    #[test]
    fn test_manual_machine_simple() {
        let term = Term::Constant {
            value: Constant::Integer(42.into()).into(),
            uniq_id: 0,
        };
        
        let mut machine = ManualMachine::new(
            Language::PlutusV2,
            CostModel::default(),
            ExBudget::default(),
            1000,
            term,
        ).unwrap();

        // Check initial state
        assert!(machine.is_ready());
        
        // Execute one step
        let _status = machine.step();
        
        // Should be ready for the next step
        assert!(machine.is_ready());
        
        // Execute until completion
        let final_status = machine.run_to_completion();
        
        // Check result
        if let ExecutionStatus::Done(result) = final_status {
            assert_eq!(result, &Term::Constant {
                value: Constant::Integer(42.into()).into(),
                uniq_id: 0,
            });
        } else {
            panic!("Expected Done status, got: {:?}", final_status);
        }
    }

    #[test]
    fn test_manual_machine_step_by_step() {
        let program: Program<NamedDeBruijn> = Program {
            version: (0, 0, 0),
            term: Term::Apply {
                uniq_id: 0,
                function: Term::Apply {
                    uniq_id: 1,
                    function: Term::Builtin {
                        fun: DefaultFunction::AddInteger,
                        uniq_id: 2,
                    }.into(),
                    argument: Term::Constant {
                        value: Constant::Integer(2.into()).into(),
                        uniq_id: 3,
                    }.into(),
                }
                .into(),
                argument: Term::Constant {
                    value: Constant::Integer(3.into()).into(),
                    uniq_id: 4,
                }.into(),
            },
        };

        let mut machine = ManualMachine::new(
            Language::PlutusV2,
            CostModel::default(),
            ExBudget::max(),
            1000,
            program.term,
        ).unwrap();

        let mut step_count = 0;
        
        // Execute step-by-step
        while machine.is_ready() {
            machine.step();
            step_count += 1;
            
            // Avoid infinite loop
            if step_count > 100 {
                panic!("Too many steps, possible infinite loop");
            }
        }

        // Check result
        if let ExecutionStatus::Done(result) = machine.status() {
            assert_eq!(result, &Term::Constant {
                value: Constant::Integer(5.into()).into(),
                uniq_id: 0,
            });
        } else {
            panic!("Expected Done status, got: {:?}", machine.status());
        }
        
        println!("Execution completed in {} steps", step_count);
    }

    #[test]
    fn test_manual_machine_reset() {
        let term1 = Term::Constant {
            value: Constant::Integer(42.into()).into(),
            uniq_id: 0,
        };
        let term2 = Term::Constant {
            value: Constant::Integer(24.into()).into(),
            uniq_id: 1,
        };
        
        let mut machine = ManualMachine::new(
            Language::PlutusV2,
            CostModel::default(),
            ExBudget::max(),
            1000,
            term1,
        ).unwrap();

        // Execute first term
        machine.run_to_completion();
        assert!(machine.is_done());

        // Reset with new term
        machine.reset(term2).unwrap();
        assert!(machine.is_ready());

        // Execute second term
        machine.run_to_completion();
        
        if let ExecutionStatus::Done(result) = machine.status() {
            assert_eq!(result, &Term::Constant {
                value: Constant::Integer(24.into()).into(),
                uniq_id: 1,
            });
        } else {
            panic!("Expected Done status after reset");
        }
    }

    // Adapted tests from machine.rs

    #[test]
    fn test_manual_machine_add_big_ints() {
        let program: Program<NamedDeBruijn> = Program {
            version: (0, 0, 0),
            term: Term::Apply {
                uniq_id: 0,
                function: Term::Apply {
                    uniq_id: 1,
                    function: Term::Builtin {
                        fun: DefaultFunction::AddInteger,
                        uniq_id: 2,
                    }.into(),
                    argument: Term::Constant {
                        value: Constant::Integer(i128::MAX.into()).into(),
                        uniq_id: 3,
                    }.into(),
                }
                .into(),
                argument: Term::Constant {
                    value: Constant::Integer(i128::MAX.into()).into(),
                    uniq_id: 4,
                }.into(),
            },
        };

        let mut machine = ManualMachine::new(
            Language::PlutusV2,
            CostModel::default(),
            ExBudget::default(),
            1000,
            program.term,
        ).unwrap();

        // Execute until completion
        machine.run_to_completion();

        // Check result
        if let ExecutionStatus::Done(result) = machine.status() {
            let expected = Term::Constant {
                value: Constant::Integer(
                    Into::<BigInt>::into(i128::MAX) + Into::<BigInt>::into(i128::MAX)
                )
                .into(),
                uniq_id: 0,
            };
            assert_eq!(result, &expected);
        } else {
            panic!("Expected Done status, got: {:?}", machine.status());
        }
    }

    #[test]
    fn test_manual_machine_divide_integer() {
        let make_program = |fun: DefaultFunction, n: i32, m: i32| Program::<NamedDeBruijn> {
            version: (0, 0, 0),
            term: Term::Apply {
                uniq_id: 0,
                function: Term::Apply {
                    uniq_id: 1,
                    function: Term::Builtin {
                        fun,
                        uniq_id: 2,
                    }.into(),
                    argument: Term::Constant {
                        value: Constant::Integer(n.into()).into(),
                        uniq_id: 3,
                    }.into(),
                }
                .into(),
                argument: Term::Constant {
                    value: Constant::Integer(m.into()).into(),
                    uniq_id: 4,
                }.into(),
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

        for (fun, n, m, expected_result) in test_data {
            let program = make_program(fun, n, m);
            
            let mut machine = ManualMachine::new(
                Language::PlutusV2,
                CostModel::default(),
                ExBudget::default(),
                1000,
                program.term,
            ).unwrap();

            // Execute with step counting
            let mut step_count = 0;
            while machine.is_ready() {
                machine.step();
                step_count += 1;
                if step_count > 100 { // protection against infinite loop
                    panic!("Too many steps for {:?}({}, {})", fun, n, m);
                }
            }

            // Check result
            if let ExecutionStatus::Done(result) = machine.status() {
                let expected = Term::Constant {
                    value: Constant::Integer(expected_result.into()).into(),
                    uniq_id: 0,
                };
                assert_eq!(result, &expected, "Failed for {:?}({}, {})", fun, n, m);
            } else {
                panic!("Expected Done status for {:?}({}, {}), got: {:?}", fun, n, m, machine.status());
            }
        }
    }

    #[test]
    fn test_manual_machine_case_constr_case_0() {
        let make_program =
            |fun: DefaultFunction, tag: usize, n: i32, m: i32| Program::<NamedDeBruijn> {
                version: (0, 0, 0),
                term: Term::Case {
                    uniq_id: 0,
                    constr: Term::Constr {
                        tag,
                        fields: vec![
                            Term::Constant {
                                value: Constant::Integer(n.into()).into(),
                                uniq_id: 1,
                            },
                            Term::Constant {
                                value: Constant::Integer(m.into()).into(),
                                uniq_id: 2,
                            },
                        ],
                        uniq_id: 3,
                    }
                    .into(),
                    branches: vec![
                        Term::Builtin {
                            fun,
                            uniq_id: 4,
                        },
                        Term::Builtin {
                            fun: DefaultFunction::SubtractInteger,
                            uniq_id: 5,
                        },
                    ],
                },
            };

        let test_data = vec![
            (DefaultFunction::AddInteger, 0, 8, 3, 11),
            (DefaultFunction::AddInteger, 1, 8, 3, 5),
        ];

        for (fun, tag, n, m, expected_result) in test_data {
            let program = make_program(fun, tag, n, m);
            
            let mut machine = ManualMachine::new(
                Language::PlutusV2,
                CostModel::default(),
                ExBudget::max(),
                1000,
                program.term,
            ).unwrap();

            // Execute step-by-step
            let mut step_count = 0;
            while machine.is_ready() {
                machine.step();
                step_count += 1;
                
                if step_count > 100 {
                    panic!("Too many steps for case test");
                }
            }

            // Check result
            if let ExecutionStatus::Done(result) = machine.status() {
                let expected = Term::Constant {
                    value: Constant::Integer(expected_result.into()).into(),
                    uniq_id: 0,
                };
                assert_eq!(result, &expected, "Failed for tag {} with {:?}", tag, fun);
            } else {
                panic!("Expected Done status for case test, got: {:?}", machine.status());
            }
        }
    }

    #[test]
    fn test_manual_machine_case_constr_case_1() {
        let make_program = |tag: usize| Program::<NamedDeBruijn> {
            version: (0, 0, 0),
            term: Term::Case {
                uniq_id: 0,
                constr: Term::Constr {
                    tag,
                    fields: vec![],
                    uniq_id: 1,
                }
                .into(),
                branches: vec![
                    Term::Constant {
                        value: Constant::Integer(5.into()).into(),
                        uniq_id: 2,
                    },
                    Term::Constant {
                        value: Constant::Integer(10.into()).into(),
                        uniq_id: 3,
                    },
                    Term::Constant {
                        value: Constant::Integer(15.into()).into(),
                        uniq_id: 4,
                    },
                ],
            },
        };

        let test_data = vec![(0, 5), (1, 10), (2, 15)];

        for (tag, expected_result) in test_data {
            let program = make_program(tag);
            
            let mut machine = ManualMachine::new(
                Language::PlutusV2,
                CostModel::default(),
                ExBudget::max(),
                1000,
                program.term,
            ).unwrap();

            // Execute 2 steps at a time
            let mut total_steps = 0;
            while machine.is_ready() {
                let steps_to_take = std::cmp::min(2, 10); // maximum 2 steps at a time
                machine.step_n(steps_to_take);
                total_steps += steps_to_take;
                
                if total_steps > 50 {
                    panic!("Too many steps for simple case test");
                }
            }

            // Check result
            if let ExecutionStatus::Done(result) = machine.status() {
                let expected = Term::Constant {
                    value: Constant::Integer(expected_result.into()).into(),
                    uniq_id: 0,
                };
                assert_eq!(result, &expected, "Failed for tag {}", tag);
            } else {
                panic!("Expected Done status for tag {}, got: {:?}", tag, machine.status());
            }
        }
    }

    #[test]
    fn test_manual_machine_step_by_step_comparison() {
        // Compare the result of ManualMachine with the regular Machine through Program::eval
        let term = Term::   Apply {
            uniq_id: 0,
            function: Term::Apply {
                uniq_id: 1,
                function: Term::Builtin {
                    fun: DefaultFunction::MultiplyInteger,
                    uniq_id: 2,
                }.into(),
                argument: Term::Constant {
                    value: Constant::Integer(6.into()).into(),
                    uniq_id: 3,
                }.into(),
            }
            .into(),
            argument: Term::Constant {
                value: Constant::Integer(7.into()).into(),
                uniq_id: 4,
            }.into(),
        };

        let program: Program<NamedDeBruijn> = Program {
            version: (0, 0, 0),
            term: term.clone(),
        };

        // Execute through the regular Machine
        let eval_result = program.eval(ExBudget::max());
        let expected_result = eval_result.result().unwrap();

        // Execute through ManualMachine
        let mut machine = ManualMachine::new(
            Language::PlutusV2,
            CostModel::default(),
            ExBudget::max(),
            1000,
            term,
        ).unwrap();

        let mut step_count = 0;
        while machine.is_ready() {
            step_count += 1;
            machine.step();
            
            if step_count > 100 {
                panic!("Too many steps");
            }
        }

        // Compare results
        if let ExecutionStatus::Done(manual_result) = machine.status() {
            assert_eq!(manual_result, &expected_result, "ManualMachine result differs from Machine result");
            assert_eq!(manual_result, &Term::Constant {
                value: Constant::Integer(42.into()).into(),
                uniq_id: 0,
            });
            println!("✓ Both machines produced the same result in {} steps", step_count);
        } else {
            panic!("ManualMachine failed: {:?}", machine.status());
        }
    }

    #[test]
    fn test_collect_nested_contexts() {
        // Create a complex nested term that will produce nested contexts
        let term = Term::Apply {
            uniq_id: 0,
            function: Term::Apply {
                uniq_id: 1,
                function: Term::Force {
                    uniq_id: 2,
                    body: Term::Delay {
                        uniq_id: 3,
                        body: Term::Builtin {
                            fun: DefaultFunction::AddInteger,
                            uniq_id: 4,
                        }.into(),
                    }.into(),
                }.into(),
                argument: Term::Constant {
                    value: Constant::Integer(10.into()).into(),
                    uniq_id: 5,
                }.into(),
            }
            .into(),
            argument: Term::Constant {
                value: Constant::Integer(20.into()).into(),
                uniq_id: 6,
            }.into(),
        };

        let mut machine = ManualMachine::new(
            Language::PlutusV2,
            CostModel::default(),
            ExBudget::max(),
            1000,
            term,
        ).unwrap();

        // Initial state should have minimal context
        let initial_contexts = machine.collect_nested_contexts();
        println!("Initial contexts count: {}", initial_contexts.len());
        assert_eq!(initial_contexts.len(), 1); // Should have at least NoFrame

        // Execute a few steps to build up context stack
        for step in 0..5 {
            if !machine.is_ready() {
                break;
            }
            
            machine.step();
            let contexts = machine.collect_nested_contexts();
            println!("Step {}: contexts count = {}", step + 1, contexts.len());
            
            // Verify we can extract contexts without panicking
            assert!(!contexts.is_empty() || machine.is_done());
            
            // Print context types for debugging
            for (i, context) in contexts.iter().enumerate() {
                let context_type = match context {
                    Context::NoFrame => "NoFrame",
                    Context::FrameForce(_) => "FrameForce",
                    Context::FrameAwaitFunTerm(_, _, _) => "FrameAwaitFunTerm",
                    Context::FrameAwaitArg(_, _) => "FrameAwaitArg",
                    Context::FrameAwaitFunValue(_, _) => "FrameAwaitFunValue",
                    Context::FrameConstr(_, _, _, _, _, _) => "FrameConstr",
                    Context::FrameCases(_, _, _) => "FrameCases",
                };
                println!("  Context[{}]: {}", i, context_type);
            }
        }

        // Execute to completion
        let final_status = machine.run_to_completion();
        
        // Check final result
        if let ExecutionStatus::Done(result) = final_status {
            println!("Final result: {:?}", result);
            // Final state should have no contexts (or just NoFrame if it's a Done state with no context)
            let final_contexts = machine.collect_nested_contexts();
            println!("Final contexts count: {}", final_contexts.len());
            // Done state returns empty list
            assert_eq!(final_contexts.len(), 0);
        } else {
            panic!("Expected Done status, got: {:?}", final_status);
        }
    }
} 