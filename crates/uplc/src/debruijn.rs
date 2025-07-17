use std::rc::Rc;

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

    pub fn name_to_named_debruijn(
        &mut self,
        term: &Term<Name>,
    ) -> Result<Term<NamedDeBruijn>, Error> {
        let converted_term = match term {
            Term::Var { name, uniq_id } => Term::Var {
                name: NamedDeBruijn {
                    text: name.text.to_string(),
                    index: self.get_index(name)?,
                }
                .into(), 
                uniq_id: *uniq_id,
            },
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.name_to_named_debruijn(body)?),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => {
                self.declare_unique(parameter_name.unique);

                let index = self.get_index(parameter_name)?;

                let name = NamedDeBruijn {
                    text: parameter_name.text.to_string(),
                    index,
                };

                self.start_scope();

                let body = self.name_to_named_debruijn(body)?;

                self.end_scope();

                self.remove_unique(parameter_name.unique);

                Term::Lambda {
                    parameter_name: name.into(),
                    body: Rc::new(body),
                    uniq_id: *uniq_id,
                }
            }
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.name_to_named_debruijn(function)?),
                argument: Rc::new(self.name_to_named_debruijn(argument)?),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.name_to_named_debruijn(body)?),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.name_to_named_debruijn(field))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.name_to_named_debruijn(constr)?),
                branches: branches
                    .iter()
                    .map(|branch| self.name_to_named_debruijn(branch))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
        };

        Ok(converted_term)
    }

    pub fn name_to_debruijn(&mut self, term: &Term<Name>) -> Result<Term<DeBruijn>, Error> {
        let converted_term = match term {
            Term::Var { name, uniq_id } => Term::Var {
                name: self.get_index(name)?.into(),
                uniq_id: *uniq_id,
            },
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.name_to_debruijn(body)?),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => {
                self.declare_unique(parameter_name.unique);

                let name = self.get_index(parameter_name)?;

                self.start_scope();

                let body = self.name_to_debruijn(body)?;

                self.end_scope();

                self.remove_unique(parameter_name.unique);

                Term::Lambda {
                    parameter_name: name.into(),
                    body: Rc::new(body),
                    uniq_id: *uniq_id,
                }
            }
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.name_to_debruijn(function)?),
                argument: Rc::new(self.name_to_debruijn(argument)?),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.name_to_debruijn(body)?),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.name_to_debruijn(field))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.name_to_debruijn(constr)?),
                branches: branches
                    .iter()
                    .map(|branch| self.name_to_debruijn(branch))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
        };

        Ok(converted_term)
    }

    pub fn named_debruijn_to_name(
        &mut self,
        term: &Term<NamedDeBruijn>,
    ) -> Result<Term<Name>, Error> {
        let converted_term = match term {
            Term::Var { name, uniq_id } => Term::Var {
                name: Name {
                    text: name.text.to_string(),
                    unique: self.get_unique(&name.index)?,
                }
                .into(),
                uniq_id: *uniq_id,
            },
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.named_debruijn_to_name(body)?),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => {
                self.declare_binder();

                let unique = self.get_unique(&parameter_name.index)?;

                let name = Name {
                    text: parameter_name.text.to_string(),
                    unique,
                };

                self.start_scope();

                let body = self.named_debruijn_to_name(body)?;

                self.end_scope();

                Term::Lambda {
                    parameter_name: name.into(),
                    body: Rc::new(body),
                    uniq_id: *uniq_id,
                }
            }
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.named_debruijn_to_name(function)?),
                argument: Rc::new(self.named_debruijn_to_name(argument)?),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.named_debruijn_to_name(body)?),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.named_debruijn_to_name(field))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.named_debruijn_to_name(constr)?),
                branches: branches
                    .iter()
                    .map(|branch| self.named_debruijn_to_name(branch))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
        };

        Ok(converted_term)
    }

    pub fn debruijn_to_name(&mut self, term: &Term<DeBruijn>) -> Result<Term<Name>, Error> {
        let converted_term = match term {
            Term::Var { name, uniq_id } => {
                let unique = self.get_unique(&name)?;

                Term::Var {
                    name: Name {
                        text: format!("i_{unique}"),
                        unique,
                    }
                    .into(),
                    uniq_id: *uniq_id,
                }
            }
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.debruijn_to_name(body)?),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => {
                self.declare_binder();

                let unique = self.get_unique(parameter_name)?;

                let name = Name {
                    text: format!("i_{unique}"),
                    unique,
                };

                self.start_scope();

                let body = self.debruijn_to_name(body)?;

                self.end_scope();

                Term::Lambda {
                    parameter_name: name.into(),
                    body: Rc::new(body),
                    uniq_id: *uniq_id,
                }
            }
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.debruijn_to_name(function)?),
                argument: Rc::new(self.debruijn_to_name(argument)?),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.debruijn_to_name(body)?),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.debruijn_to_name(field))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.debruijn_to_name(constr)?),
                branches: branches
                    .iter()
                    .map(|branch| self.debruijn_to_name(branch))
                    .collect::<Result<_, _>>()?,
                uniq_id: *uniq_id,
            },
        };

        Ok(converted_term)
    }

    #[allow(clippy::only_used_in_recursion)]
    pub fn named_debruijn_to_debruijn(&mut self, term: &Term<NamedDeBruijn>) -> Term<DeBruijn> {
        match term {
            Term::Var { name, uniq_id } => Term::Var {
                name: name.index.into(),
                uniq_id: *uniq_id,
            },
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.named_debruijn_to_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => Term::Lambda {
                parameter_name: parameter_name.index.into(),
                body: Rc::new(self.named_debruijn_to_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.named_debruijn_to_debruijn(function)),
                argument: Rc::new(self.named_debruijn_to_debruijn(argument)),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.named_debruijn_to_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.named_debruijn_to_debruijn(field))
                    .collect(),
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.named_debruijn_to_debruijn(constr)),
                branches: branches
                    .iter()
                    .map(|branch| self.named_debruijn_to_debruijn(branch))
                    .collect(),
                uniq_id: *uniq_id,
            },
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    pub fn debruijn_to_named_debruijn(&mut self, term: &Term<DeBruijn>) -> Term<NamedDeBruijn> {
        match term {
            Term::Var { name, uniq_id } => Term::Var {
                name: NamedDeBruijn {
                    text: "i".to_string(),
                    index: *name.as_ref(),
                }
                .into(),
                uniq_id: *uniq_id,
            },
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.debruijn_to_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => Term::Lambda {
                parameter_name: NamedDeBruijn::from(*parameter_name.as_ref()).into(),
                body: Rc::new(self.debruijn_to_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.debruijn_to_named_debruijn(function)),
                argument: Rc::new(self.debruijn_to_named_debruijn(argument)),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.debruijn_to_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.debruijn_to_named_debruijn(field))
                    .collect(),
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.debruijn_to_named_debruijn(constr)),
                branches: branches
                    .iter()
                    .map(|branch| self.debruijn_to_named_debruijn(branch))
                    .collect(),
                uniq_id: *uniq_id,
            },
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    pub fn fake_named_debruijn_to_named_debruijn(
        &mut self,
        term: &Term<FakeNamedDeBruijn>,
    ) -> Term<NamedDeBruijn> {
        match term {
            Term::Var { name, uniq_id } => Term::Var {
                name: NamedDeBruijn::from(name.as_ref().clone()).into(),
                uniq_id: *uniq_id,
            },
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.fake_named_debruijn_to_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => Term::Lambda {
                parameter_name: NamedDeBruijn::from(parameter_name.as_ref().clone()).into(),
                body: Rc::new(self.fake_named_debruijn_to_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.fake_named_debruijn_to_named_debruijn(function)),
                argument: Rc::new(self.fake_named_debruijn_to_named_debruijn(argument)),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.fake_named_debruijn_to_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.fake_named_debruijn_to_named_debruijn(field))
                    .collect(),
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.fake_named_debruijn_to_named_debruijn(constr)),
                branches: branches
                    .iter()
                    .map(|branch| self.fake_named_debruijn_to_named_debruijn(branch))
                    .collect(),
                uniq_id: *uniq_id,
            },
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    pub fn named_debruijn_to_fake_named_debruijn(
        &mut self,
        term: &Term<NamedDeBruijn>,
    ) -> Term<FakeNamedDeBruijn> {
        match term {
            Term::Var { name, uniq_id } => Term::Var {
                name: FakeNamedDeBruijn::from(name.as_ref().clone()).into(),
                uniq_id: *uniq_id,
            },
            Term::Delay { body, uniq_id } => Term::Delay {
                body: Rc::new(self.named_debruijn_to_fake_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Lambda {
                parameter_name,
                body,
                uniq_id,
            } => Term::Lambda {
                parameter_name: FakeNamedDeBruijn::from(parameter_name.as_ref().clone()).into(),
                body: Rc::new(self.named_debruijn_to_fake_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Apply { function, argument, uniq_id } => Term::Apply {
                function: Rc::new(self.named_debruijn_to_fake_named_debruijn(function)),
                argument: Rc::new(self.named_debruijn_to_fake_named_debruijn(argument)),
                uniq_id: *uniq_id,
            },
            Term::Constant { value, uniq_id } => Term::Constant {
                value: value.clone(),
                uniq_id: *uniq_id,
            },
            Term::Force { body, uniq_id } => Term::Force {
                body: Rc::new(self.named_debruijn_to_fake_named_debruijn(body)),
                uniq_id: *uniq_id,
            },
            Term::Error { uniq_id } => Term::Error { uniq_id: *uniq_id },
            Term::Builtin { fun, uniq_id } => Term::Builtin {
                fun: *fun,
                uniq_id: *uniq_id,
            },
            Term::Constr { tag, fields, uniq_id } => Term::Constr {
                tag: *tag,
                fields: fields
                    .iter()
                    .map(|field| self.named_debruijn_to_fake_named_debruijn(field))
                    .collect(),
                uniq_id: *uniq_id,
            },
            Term::Case { constr, branches, uniq_id } => Term::Case {
                constr: Rc::new(self.named_debruijn_to_fake_named_debruijn(constr)),
                branches: branches
                    .iter()
                    .map(|branch| self.named_debruijn_to_fake_named_debruijn(branch))
                    .collect(),
                uniq_id: *uniq_id,
            },
        }
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
