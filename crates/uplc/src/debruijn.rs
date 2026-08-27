use std::{convert::Infallible, rc::Rc};

use thiserror::Error;

use crate::ast::{DeBruijn, FakeNamedDeBruijn, Name, NamedDeBruijn, Term, Unique};

mod bimap;

#[derive(Debug, Clone, PartialEq, Copy, Eq, Hash)]
struct Level(usize);

#[derive(Error, Debug)]
pub enum Error {
    #[error("Free Unique {} with name {}", .0.unique, .0.text)]
    FreeUnique(Name),
    #[error("Free Index `{0}`")]
    FreeIndex(DeBruijn),
}

pub struct Converter {
    current_level: Level,
    levels: Vec<bimap::BiMap>,
    current_unique: Unique,
}

impl Converter {
    pub fn new() -> Self {
        Converter {
            current_level: Level(0),
            levels: vec![bimap::BiMap::new()],
            current_unique: Unique::new(0),
        }
    }

    /// The one term walk, shared by every converter in this file.
    ///
    /// It is iterative because the nesting depth of a term is attacker-controlled: the cheapest
    /// level in flat is a 4-bit `delay` tag, half a byte, so a script inside the on-chain size
    /// limit is tens of thousands of levels deep, and on `wasm32` the engine's call stack is about
    /// a megabyte and cannot be grown from the page. Conversion runs on every decode --
    /// `Program<FakeNamedDeBruijn>` into `Program<NamedDeBruijn>` is the last step of turning hex
    /// into a program -- so recursing here put a ceiling on the whole path. Do not fold this back
    /// into a recursive walk.
    ///
    /// Every conversion below is the same tree map, copying the structure verbatim; they differ
    /// only in how a `Var`'s name and a `Lambda`'s parameter name are rewritten, and in the scope
    /// bookkeeping around a lambda body. Those pieces arrive as closures, so the walk -- and the
    /// depth fix -- lives in exactly one place and the conversions cannot drift apart.
    ///
    /// `enter_lambda` runs before the body and `exit_lambda` once the body is finished: the two
    /// points where the recursive version opened and closed the scope around its recursive call.
    /// A failure leaves the scope stack unbalanced, exactly as returning early out of the
    /// recursion did.
    fn convert_term<'a, A, B, E>(
        &mut self,
        term: &'a Term<A>,
        mut var: impl FnMut(&mut Self, &'a Rc<A>) -> Result<Rc<B>, E>,
        mut enter_lambda: impl FnMut(&mut Self, &'a Rc<A>) -> Result<Rc<B>, E>,
        mut exit_lambda: impl FnMut(&mut Self, &'a Rc<A>),
    ) -> Result<Term<B>, E> {
        /// A node whose children are still being converted, holding the parts already in hand.
        enum Build<'a, A, B> {
            Delay {
                uniq_id: isize,
            },
            Force {
                uniq_id: isize,
            },
            Lambda {
                parameter_name: Rc<B>,
                /// Kept so `exit_lambda` sees the binder the scope was opened for.
                source_name: &'a Rc<A>,
                uniq_id: isize,
            },
            Apply {
                uniq_id: isize,
            },
            Constr {
                tag: usize,
                arity: usize,
                uniq_id: isize,
            },
            Case {
                arity: usize,
                uniq_id: isize,
            },
        }

        enum Step<'a, A, B> {
            /// Convert this subterm.
            Convert(&'a Term<A>),
            /// Reassemble a node from the results its children left behind.
            Build(Build<'a, A, B>),
        }

        // `todo` is LIFO, so children go on in reverse of the order they are converted in. `done`
        // collects finished subterms in that same order, which is what lets each `Build` take its
        // children straight back off the end.
        let mut todo: Vec<Step<'a, A, B>> = vec![Step::Convert(term)];
        let mut done: Vec<Term<B>> = Vec::new();

        while let Some(step) = todo.pop() {
            match step {
                Step::Convert(term) => match term {
                    Term::Var { name, uniq_id } => done.push(Term::Var {
                        name: var(self, name)?,
                        uniq_id: *uniq_id,
                    }),
                    Term::Delay { body, uniq_id } => {
                        todo.push(Step::Build(Build::Delay { uniq_id: *uniq_id }));
                        todo.push(Step::Convert(body));
                    }
                    Term::Lambda {
                        parameter_name,
                        body,
                        uniq_id,
                    } => {
                        let converted_name = enter_lambda(self, parameter_name)?;

                        todo.push(Step::Build(Build::Lambda {
                            parameter_name: converted_name,
                            source_name: parameter_name,
                            uniq_id: *uniq_id,
                        }));
                        todo.push(Step::Convert(body));
                    }
                    Term::Apply {
                        function,
                        argument,
                        uniq_id,
                    } => {
                        todo.push(Step::Build(Build::Apply { uniq_id: *uniq_id }));
                        todo.push(Step::Convert(argument));
                        todo.push(Step::Convert(function));
                    }
                    Term::Constant { value, uniq_id } => done.push(Term::Constant {
                        // A constant is carried over untouched: this clones the `Rc`, not the
                        // constant behind it.
                        value: value.clone(),
                        uniq_id: *uniq_id,
                    }),
                    Term::Force { body, uniq_id } => {
                        todo.push(Step::Build(Build::Force { uniq_id: *uniq_id }));
                        todo.push(Step::Convert(body));
                    }
                    Term::Error { uniq_id } => done.push(Term::Error { uniq_id: *uniq_id }),
                    Term::Builtin { fun, uniq_id } => done.push(Term::Builtin {
                        fun: *fun,
                        uniq_id: *uniq_id,
                    }),
                    Term::Constr {
                        tag,
                        fields,
                        uniq_id,
                    } => {
                        todo.push(Step::Build(Build::Constr {
                            tag: *tag,
                            arity: fields.len(),
                            uniq_id: *uniq_id,
                        }));

                        for field in fields.iter().rev() {
                            todo.push(Step::Convert(field));
                        }
                    }
                    Term::Case {
                        constr,
                        branches,
                        uniq_id,
                    } => {
                        todo.push(Step::Build(Build::Case {
                            arity: branches.len(),
                            uniq_id: *uniq_id,
                        }));

                        for branch in branches.iter().rev() {
                            todo.push(Step::Convert(branch));
                        }

                        todo.push(Step::Convert(constr));
                    }
                },

                Step::Build(build) => match build {
                    Build::Delay { uniq_id } => {
                        let body = Rc::new(done.pop().expect("body converted"));

                        done.push(Term::Delay { body, uniq_id });
                    }
                    Build::Force { uniq_id } => {
                        let body = Rc::new(done.pop().expect("body converted"));

                        done.push(Term::Force { body, uniq_id });
                    }
                    Build::Lambda {
                        parameter_name,
                        source_name,
                        uniq_id,
                    } => {
                        let body = Rc::new(done.pop().expect("body converted"));

                        exit_lambda(self, source_name);

                        done.push(Term::Lambda {
                            parameter_name,
                            body,
                            uniq_id,
                        });
                    }
                    Build::Apply { uniq_id } => {
                        // The function was converted first, so it sits under the argument.
                        let argument = Rc::new(done.pop().expect("argument converted"));
                        let function = Rc::new(done.pop().expect("function converted"));

                        done.push(Term::Apply {
                            function,
                            argument,
                            uniq_id,
                        });
                    }
                    Build::Constr {
                        tag,
                        arity,
                        uniq_id,
                    } => {
                        let fields = done.split_off(done.len() - arity);

                        done.push(Term::Constr {
                            tag,
                            fields,
                            uniq_id,
                        });
                    }
                    Build::Case { arity, uniq_id } => {
                        let branches = done.split_off(done.len() - arity);
                        let constr = Rc::new(done.pop().expect("scrutinee converted"));

                        done.push(Term::Case {
                            constr,
                            branches,
                            uniq_id,
                        });
                    }
                },
            }
        }

        Ok(done.pop().expect("the walk leaves the converted root behind"))
    }

    /// The same walk for the conversions that only rename, and so cannot fail.
    ///
    /// `Infallible` rather than an `expect`: if one of these mappings ever grows a failure case,
    /// this stops compiling instead of turning into a panic at run time.
    fn convert_term_infallible<'a, A, B>(
        &mut self,
        term: &'a Term<A>,
        mut var: impl FnMut(&mut Self, &'a Rc<A>) -> Rc<B>,
        mut lambda: impl FnMut(&mut Self, &'a Rc<A>) -> Rc<B>,
    ) -> Term<B> {
        let converted = self.convert_term(
            term,
            |converter, name| Ok::<_, Infallible>(var(converter, name)),
            |converter, parameter_name| Ok(lambda(converter, parameter_name)),
            |_, _| {},
        );

        match converted {
            Ok(term) => term,
            Err(never) => match never {},
        }
    }

    pub fn name_to_named_debruijn(
        &mut self,
        term: &Term<Name>,
    ) -> Result<Term<NamedDeBruijn>, Error> {
        self.convert_term(
            term,
            |converter, name| {
                Ok(NamedDeBruijn {
                    text: name.text.to_string(),
                    index: converter.get_index(name)?,
                }
                .into())
            },
            |converter, parameter_name| {
                converter.declare_unique(parameter_name.unique);

                let index = converter.get_index(parameter_name)?;

                converter.start_scope();

                Ok(NamedDeBruijn {
                    text: parameter_name.text.to_string(),
                    index,
                }
                .into())
            },
            |converter, parameter_name| {
                converter.end_scope();
                converter.remove_unique(parameter_name.unique);
            },
        )
    }

    pub fn name_to_debruijn(&mut self, term: &Term<Name>) -> Result<Term<DeBruijn>, Error> {
        self.convert_term(
            term,
            |converter, name| Ok(converter.get_index(name)?.into()),
            |converter, parameter_name| {
                converter.declare_unique(parameter_name.unique);

                let index = converter.get_index(parameter_name)?;

                converter.start_scope();

                Ok(index.into())
            },
            |converter, parameter_name| {
                converter.end_scope();
                converter.remove_unique(parameter_name.unique);
            },
        )
    }

    pub fn named_debruijn_to_name(
        &mut self,
        term: &Term<NamedDeBruijn>,
    ) -> Result<Term<Name>, Error> {
        self.convert_term(
            term,
            |converter, name| {
                Ok(Name {
                    text: name.text.to_string(),
                    unique: converter.get_unique(&name.index)?,
                }
                .into())
            },
            |converter, parameter_name| {
                converter.declare_binder();

                let unique = converter.get_unique(&parameter_name.index)?;

                converter.start_scope();

                Ok(Name {
                    text: parameter_name.text.to_string(),
                    unique,
                }
                .into())
            },
            |converter, _| converter.end_scope(),
        )
    }

    pub fn debruijn_to_name(&mut self, term: &Term<DeBruijn>) -> Result<Term<Name>, Error> {
        self.convert_term(
            term,
            |converter, name| {
                let unique = converter.get_unique(name)?;

                Ok(Name {
                    text: format!("i_{unique}"),
                    unique,
                }
                .into())
            },
            |converter, parameter_name| {
                converter.declare_binder();

                let unique = converter.get_unique(parameter_name)?;

                converter.start_scope();

                Ok(Name {
                    text: format!("i_{unique}"),
                    unique,
                }
                .into())
            },
            |converter, _| converter.end_scope(),
        )
    }

    pub fn named_debruijn_to_debruijn(&mut self, term: &Term<NamedDeBruijn>) -> Term<DeBruijn> {
        self.convert_term_infallible(
            term,
            |_, name| name.index.into(),
            |_, parameter_name| parameter_name.index.into(),
        )
    }

    pub fn debruijn_to_named_debruijn(&mut self, term: &Term<DeBruijn>) -> Term<NamedDeBruijn> {
        self.convert_term_infallible(
            term,
            |_, name| {
                NamedDeBruijn {
                    text: "i".to_string(),
                    index: *name.as_ref(),
                }
                .into()
            },
            |_, parameter_name| NamedDeBruijn::from(*parameter_name.as_ref()).into(),
        )
    }

    pub fn fake_named_debruijn_to_named_debruijn(
        &mut self,
        term: &Term<FakeNamedDeBruijn>,
    ) -> Term<NamedDeBruijn> {
        self.convert_term_infallible(
            term,
            |_, name| NamedDeBruijn::from(name.as_ref().clone()).into(),
            |_, parameter_name| NamedDeBruijn::from(parameter_name.as_ref().clone()).into(),
        )
    }

    pub fn named_debruijn_to_fake_named_debruijn(
        &mut self,
        term: &Term<NamedDeBruijn>,
    ) -> Term<FakeNamedDeBruijn> {
        self.convert_term_infallible(
            term,
            |_, name| FakeNamedDeBruijn::from(name.as_ref().clone()).into(),
            |_, parameter_name| FakeNamedDeBruijn::from(parameter_name.as_ref().clone()).into(),
        )
    }

    fn get_index(&mut self, name: &Name) -> Result<DeBruijn, Error> {
        for scope in self.levels.iter().rev() {
            if let Some(found_level) = scope.get(&name.unique) {
                let index = self.current_level.0 - found_level.0;

                return Ok(index.into());
            }
        }

        Err(Error::FreeUnique(name.clone()))
    }

    fn get_unique(&mut self, index: &DeBruijn) -> Result<Unique, Error> {
        for scope in self.levels.iter().rev() {
            let index = Level(
                self.current_level
                    .0
                    .checked_sub(index.inner())
                    .ok_or(Error::FreeIndex(*index))?,
            );

            if let Some(unique) = scope.get_right(&index) {
                return Ok(*unique);
            }
        }

        Err(Error::FreeIndex(*index))
    }

    fn declare_unique(&mut self, unique: Unique) {
        let scope = &mut self.levels[self.current_level.0];

        scope.insert(unique, self.current_level);
    }

    fn remove_unique(&mut self, unique: Unique) {
        let scope = &mut self.levels[self.current_level.0];

        scope.remove(unique, self.current_level);
    }

    fn declare_binder(&mut self) {
        let scope = &mut self.levels[self.current_level.0];

        scope.insert(self.current_unique, self.current_level);

        self.current_unique.increment();
    }

    fn start_scope(&mut self) {
        self.current_level = Level(self.current_level.0 + 1);

        self.levels.push(bimap::BiMap::new());
    }

    fn end_scope(&mut self) {
        self.current_level = Level(self.current_level.0 - 1);

        self.levels.pop();
    }
}
