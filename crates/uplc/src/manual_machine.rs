use std::ops::{Deref, DerefMut};

use crate::ast::{NamedDeBruijn, Term};
use crate::machine::{
    Context, Error, Machine, MachineState,
    cost_model::{CostModel, ExBudget},
};
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

/// A CEK machine driven one transition at a time.
///
/// Every transition is delegated to [`Machine`] itself, so stepping a program to completion
/// yields exactly the result, budget and traces `Machine::run` does for the same language,
/// protocol version and cost model. The wrapped machine is reachable through `Deref`, which keeps
/// its public fields (`ex_budget`, `traces`, `spend_counter`) addressable on a `ManualMachine`.
#[derive(Clone, Debug)]
pub struct ManualMachine {
    /// The machine every transition is delegated to (budget, traces, semantics, cost model)
    machine: Machine,
    /// Current machine state
    state: MachineState,
    /// Execution status
    status: ExecutionStatus,
    /// uniq_id of the most recent term entered in a Compute step. After completion the Done state
    /// holds a discharged value (synthetic id), so this is the last REAL source term that ran —
    /// used to point "where execution finished" at actual code on a successful run.
    last_term_id: isize,
}

impl Deref for ManualMachine {
    type Target = Machine;

    fn deref(&self) -> &Machine {
        &self.machine
    }
}

impl DerefMut for ManualMachine {
    fn deref_mut(&mut self) -> &mut Machine {
        &mut self.machine
    }
}

impl ManualMachine {
    /// Creates a new ManualMachine with the default builtin semantics of `version` (as
    /// `Machine::new`).
    pub fn new(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
        term: Term<NamedDeBruijn>,
    ) -> Result<Self, Error> {
        Self::start(
            Machine::new(version, costs, initial_budget, slippage),
            term,
        )
    }

    /// Creates a new ManualMachine whose builtin semantics follow `version` at the given
    /// protocol major version (as `Machine::new_with_protocol`).
    pub fn new_with_protocol(
        version: Language,
        protocol_major_version: u16,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
        term: Term<NamedDeBruijn>,
    ) -> Result<Self, Error> {
        Self::start(
            Machine::new_with_protocol(
                version,
                protocol_major_version,
                costs,
                initial_budget,
                slippage,
            ),
            term,
        )
    }

    /// Creates a new ManualMachine with debug information
    pub fn new_debug(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
        term: Term<NamedDeBruijn>,
    ) -> Result<Self, Error> {
        Self::start(
            Machine::new_debug(version, costs, initial_budget, slippage),
            term,
        )
    }

    /// Creates a new ManualMachine with debug information, at the given protocol major version.
    pub fn new_debug_with_protocol(
        version: Language,
        protocol_major_version: u16,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
        term: Term<NamedDeBruijn>,
    ) -> Result<Self, Error> {
        Self::start(
            Machine::new_debug_with_protocol(
                version,
                protocol_major_version,
                costs,
                initial_budget,
                slippage,
            ),
            term,
        )
    }

    /// Wraps an already configured machine. The program is validated against the machine's
    /// semantics and the startup cost is charged, exactly as `Machine::run` does before its first
    /// step.
    pub fn from_machine(machine: Machine, term: Term<NamedDeBruijn>) -> Result<Self, Error> {
        Self::start(machine, term)
    }

    fn start(mut machine: Machine, term: Term<NamedDeBruijn>) -> Result<Self, Error> {
        let state = machine.start(term)?;

        Ok(ManualMachine {
            machine,
            state,
            status: ExecutionStatus::Ready,
            last_term_id: -1,
        })
    }

    /// The machine every transition is delegated to.
    pub fn machine(&self) -> &Machine {
        &self.machine
    }

    /// Returns the current execution status
    pub fn status(&self) -> &ExecutionStatus {
        &self.status
    }

    /// Returns the current machine state
    pub fn current_state(&self) -> &MachineState {
        &self.state
    }

    /// uniq_id of the last term entered in a Compute step (the last real source term that ran),
    /// or -1 if none. Useful for pointing at "where execution finished" once the machine is Done.
    pub fn last_term_id(&self) -> isize {
        self.last_term_id
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
        // when a step errors — so it must be a STABLE sentinel, not `next_uniq_id()`: a fresh
        // global id here would burn one id per step and, because the global counter never resets
        // in a long-lived host, make the error term's id differ every run. -1 is the "no specific
        // term" sentinel and is fully deterministic.
        match std::mem::replace(&mut self.state, MachineState::Done(Term::Error { uniq_id: -1 })) {
            MachineState::Compute(context, env, term) => {
                // Capture the id of the term being computed before it is moved, so on failure the
                // resulting Error state names the actual source term that failed (e.g. a source
                // `(error)`) rather than the -1 sentinel; also record it as the last real term run.
                let failing_id = term.uniq_id();
                self.last_term_id = failing_id;
                match self.machine.compute(context, env, term) {
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
                match self.machine.return_compute(context, value) {
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
        while let ExecutionStatus::Ready = self.step() {}

        &self.status
    }

    /// Executes a given number of steps
    pub fn step_n(&mut self, n: usize) -> &ExecutionStatus {
        for _ in 0..n {
            if !matches!(self.step(), ExecutionStatus::Ready) {
                break;
            }
        }

        &self.status
    }

    /// Resets the machine with a new term, keeping the remaining budget.
    pub fn reset(&mut self, term: Term<NamedDeBruijn>) -> Result<(), Error> {
        let startup_budget = self.machine.startup_budget();

        // Check if we have enough budget for the new run
        if self.machine.ex_budget.mem < startup_budget.mem
            || self.machine.ex_budget.cpu < startup_budget.cpu
        {
            return Err(Error::OutOfExError(self.machine.ex_budget));
        }

        if !self.machine.semantics().supports_values() {
            Machine::assert_no_values(&term)?;
        }

        self.machine.clear_run_state();
        self.state = self.machine.start(term)?;
        self.status = ExecutionStatus::Ready;
        self.last_term_id = -1;

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

    #[test]
    fn test_manual_machine_follows_protocol_version() {
        let term = || Term::Builtin {
            fun: DefaultFunction::LookupCoin,
            uniq_id: 0,
        };

        // The value builtins only exist for Plutus V3 from protocol version 11 onwards, and the
        // program is refused up front, exactly as `Machine::run` refuses it.
        let refused = ManualMachine::new_with_protocol(
            Language::PlutusV3,
            10,
            CostModel::default_for_language_and_protocol(&Language::PlutusV3, 10),
            ExBudget::max(),
            200,
            term(),
        );

        assert!(matches!(refused, Err(Error::BuiltinNotAvailable(_))));

        let mut machine = ManualMachine::new_with_protocol(
            Language::PlutusV3,
            11,
            CostModel::default_for_language_and_protocol(&Language::PlutusV3, 11),
            ExBudget::max(),
            200,
            term(),
        )
        .unwrap();

        assert!(matches!(machine.run_to_completion(), ExecutionStatus::Done(_)));
    }
} 