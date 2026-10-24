use super::{ExBudget, Value};
use crate::{
    ast::{NamedDeBruijn, Term, Type, ValueError},
    builtins::DefaultFunction,
};
use num_bigint::BigInt;
use std::{
    fmt::{self, Write},
    string::FromUtf8Error,
};

#[derive(Debug, Clone, PartialEq, thiserror::Error, miette::Diagnostic)]
pub enum Error {
    #[error("execution went over budget\n{:>13} {}\n{:>13} {}", "Mem", .0.mem, "CPU", .0.cpu)]
    OutOfExError(ExBudget),
    #[error("invalid step kind: {0}")]
    InvalidStepKind(u8),
    #[error(
        "cannot evaluate an open term:\n{:>13} {}",
        "Term",
        indent(redacted(.0.to_pretty(), 10)),
    )]
    OpenTermEvaluated(Term<NamedDeBruijn>),
    #[error("the validator crashed / exited prematurely")]
    EvaluationFailure,
    #[error(
        "attempted to instantiate a non-polymorphic term\n{:>13} {}",
        "Term",
        indent(redacted_debug(&.0, 10)),
    )]
    NonPolymorphicInstantiation(Value),
    #[error(
      "attempted to apply an argument to a non-function\n{:>13} {}\n{:>13} {}",
      "Thing",
      indent(redacted_debug(&.0, 5)),
      "Argument",
      indent(redacted_debug(&.1, 5)),
    )]
    NonFunctionalApplication(Value, Value),
    #[error(
        "attempted to case a non-const\n{:>13} {}",
        "Value",
        indent(redacted_debug(&.0, 10)),
    )]
    NonConstrScrutinized(Value),
    #[error("Cases: {0:#?}\n\n are missing branch for constr:\n\n{1:#?}")]
    MissingCaseBranch(Vec<Term<NamedDeBruijn>>, Value),
    #[error("type mismatch\n{:>13} {0}\n{:>13} {1}", "Expected", "Got")]
    TypeMismatch(Type, Type),
    #[error("type mismatch\n{:>13} (list a)\n{:>13} {0}", "Expected", "Got")]
    ListTypeMismatch(Type),
    #[error("type mismatch\n{:>13} (array a)\n{:>13} {0}", "Expected", "Got")]
    ArrayTypeMismatch(Type),
    #[error("type mismatch\n{:>13}(pair a b)\n{:>13} {0}", "Expected", "Got")]
    PairTypeMismatch(Type),
    #[error(
        "unexpected empty list\n{:>13} {}",
        "List",
        indent(redacted_debug(&.0, 10)),
    )]
    EmptyList(Value),
    #[error(
        "a builtin received a term argument when something else was expected\n{:>13} {}\n{:>13} You probably forgot to wrap the builtin with a force.",
        "Term",
        indent(redacted_debug(&.0, 10)),
        "Hint"
    )]
    UnexpectedBuiltinTermArgument(Term<NamedDeBruijn>),
    #[error(
        "a builtin expected a term argument, but something else was received:\n{:>13} {}\n{:>13} You probably have an extra force wrapped around a builtin",
        "Term",
        indent(redacted_debug(&.0, 10)),
        "Hint"
    )]
    BuiltinTermArgumentExpected(Term<NamedDeBruijn>),
    #[error(
        "Unable to unlift value because it is not a constant\n{:>13} {}",
        "Value",
        indent(redacted_debug(&.0, 10)),
    )]
    NotAConstant(Value),
    #[error("The evaluation never reached a final state")]
    MachineNeverReachedDone,
    #[error("integerToByteString encountered negative size\n{:>13} {0}", "Size")]
    IntegerToByteStringNegativeSize(BigInt),
    #[error("replicateByte encountered negative size\n{:>13} {0}", "Size")]
    ReplicateByteNegativeSize(BigInt),
    #[error("integerToByteString encountered negative input\n{:>13} {0}", "Input")]
    IntegerToByteStringNegativeInput(BigInt),
    #[error(
        "bytes size beyond limit when converting from integer\n{:>13} {0}\n{:>13} {1}",
        "Size",
        "Maximum"
    )]
    IntegerToByteStringSizeTooBig(BigInt, i64),
    #[error(
        "bytes size beyond limit when replicate byte\n{:>13} {0}\n{:>13} {1}",
        "Size",
        "Maximum"
    )]
    ReplicateByteSizeTooBig(BigInt, i64),
    #[error(
        "bytes size below limit when converting from integer\n{:>13} {0}\n{:>13} {1}",
        "Size",
        "Minimum"
    )]
    IntegerToByteStringSizeTooSmall(BigInt, usize),
    #[error("Decoding utf8")]
    Utf8(#[from] FromUtf8Error),
    #[error(
        "Out of Bounds\n{:>13} {}\nb{:>13} {}\n{:>13} {}",
        "Index",
        .0,
        "ByteArray",
        hex::encode(.1),
        "Allowed",
        allowed_indices(.1)
    )]
    ByteStringOutOfBounds(BigInt, Vec<u8>),
    #[error("array index out of bounds\n{:>13} {0}\n{:>13} {1}", "Index", "Length")]
    ArrayIndexOutOfBounds(BigInt, usize),
    #[error(
        "attempt to consByteString something than isn't a byte between [0-255]\n{:>13} {0}",
        "Found"
    )]
    ByteStringConsNotAByte(BigInt),
    #[error("divide By Zero: {0} / {1}")]
    DivideByZero(BigInt, BigInt),
    #[error("expModInteger has no modular inverse for base {0} modulo {1}")]
    ExpModIntegerNoInverse(BigInt, BigInt),
    #[error("Ed25519S PublicKey should be 32 bytes but it was {0}")]
    UnexpectedEd25519PublicKeyLength(usize),
    #[error("Ed25519S Signature should be 64 bytes but it was {0}")]
    UnexpectedEd25519SignatureLength(usize),
    #[error("Secp256k1 {0} should be {1} bytes but it was {2}")]
    UnexpectedSecp256k1Length(&'static str, usize, usize),
    #[error(
      "failed to deserialise PlutusData using {0}\n{:>13} {}",
      "Value",
      indent(redacted_debug(&.1, 10)),
    )]
    DeserialisationError(String, Value),
    #[error("integer overflow")]
    OverflowError,
    #[error("value builtin failed: {0}")]
    ValueBuiltin(#[source] ValueError),
    #[error("builtin {0} is not available in this plutus / protocol version")]
    BuiltinNotAvailable(DefaultFunction),
    #[error("Value constants are not available in this plutus / protocol version")]
    ValueConstantNotAvailable,
    #[error("array constants are not available in this plutus / protocol version")]
    ArrayConstantNotAvailable,
    #[error("program version {0}.{1}.{2} is not available in this plutus / protocol version")]
    ProgramVersionNotAvailable(usize, usize, usize),
    #[error("{0} terms are not available in this plutus / protocol / program version")]
    ConstrCaseNotAvailable(&'static str),
    #[error("a constr term carries {0} fields, but at most {1} are allowed")]
    ConstrTooManyFields(usize, usize),
    #[error("a constant's type spans more than {0} type tags")]
    TypeHeaderTooLong(usize),
    #[error("{0} is not within the bounds of a Natural")]
    OutsideNaturalBounds(BigInt),
    #[error("{0} is not within the bounds of a Byte")]
    OutsideByteBounds(BigInt),
    #[error("{0} is not within the bounds of a 64-bit integer")]
    OutsideInt64Bounds(BigInt),
    #[error("readBit: index out of bounds")]
    ReadBitOutOfBounds,
    #[error("writeBits: index out of bounds")]
    WriteBitsOutOfBounds,
    #[error("illegal operation on empty ByteArray")]
    EmptyByteArray,
    #[error("blst error {0:?}")]
    Blst(blst::BLST_ERROR),
    #[error("blst::hashToGroup")]
    HashToCurveDstTooBig,
    #[error("Scalar exceeds 512-byte bound for multiScalarMul")]
    MsmScalarOutOfBounds,
    #[cfg(not(target_family = "wasm"))]
    #[error(transparent)]
    Secp256k1(#[from] secp256k1::Error),
    #[cfg(target_family = "wasm")]
    #[error("{0}")]
    K256Error(String),
}

#[cfg(target_family = "wasm")]
impl From<k256::ecdsa::Error> for Error {
    fn from(error: k256::ecdsa::Error) -> Error {
        Error::K256Error(format!("K256 error: {}", error))
    }
}

/// Print only the first n lines of possibly long output, and redact the rest if longer.
fn redacted(s: String, max_rows: usize) -> String {
    let rows = s.lines();

    if rows.count() > max_rows {
        let last_line = s.lines().last().unwrap();
        let mut s = s.lines().take(max_rows).collect::<Vec<_>>().join("\n");
        s.push_str(&format!("\n    ...redacted...\n{last_line}"));
        s
    } else {
        s
    }
}

/// `redacted(format!("{value:#?}"), max_rows)`, without formatting the rows it leaves out.
///
/// The values and terms errors carry are as deep as the script makes them, and their `{:#?}` grows
/// with the square of that depth (each row is indented as deep as it is nested), so formatting all
/// of it to show a dozen rows can run out of memory. Formatting stops once it is clear that rows
/// are left out. The last row, which `redacted` keeps, is then the closing bracket of the outermost
/// `debug_struct`, `debug_tuple` or `debug_list`, matching the bracket the first row ends with.
fn redacted_debug(value: &impl fmt::Debug, max_rows: usize) -> String {
    let mut rows = FirstRows {
        text: String::new(),
        newlines: 0,
        max_newlines: max_rows + 1,
    };

    // Whether the formatting got through says no more than `is_full` does.
    let _ = write!(rows, "{value:#?}");

    if !rows.is_full() {
        return redacted(rows.text, max_rows);
    }

    let last_line = match rows
        .text
        .lines()
        .next()
        .and_then(|first| first.chars().last())
    {
        Some('{') => "}",
        Some('(') => ")",
        Some('[') => "]",
        // Not laid out by those builders: take the long way.
        _ => return redacted(format!("{value:#?}"), max_rows),
    };

    let mut s = rows
        .text
        .lines()
        .take(max_rows)
        .collect::<Vec<_>>()
        .join("\n");
    s.push_str(&format!("\n    ...redacted...\n{last_line}"));
    s
}

/// Collects what is written to it up to its `max_newlines`th newline, then fails that write and
/// every later one.
///
/// `{:#?}` output never ends with a newline, so getting that far means at least one more row.
struct FirstRows {
    text: String,
    newlines: usize,
    max_newlines: usize,
}

impl FirstRows {
    fn is_full(&self) -> bool {
        self.newlines >= self.max_newlines
    }
}

impl Write for FirstRows {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.is_full() {
            return Err(fmt::Error);
        }

        for (at, _) in s.match_indices('\n') {
            self.newlines += 1;

            if self.is_full() {
                self.text.push_str(&s[..at]);
                return Err(fmt::Error);
            }
        }

        self.text.push_str(s);

        Ok(())
    }
}

/// The indices of a byte string, as a range; an empty one has none.
fn allowed_indices(bytes: &[u8]) -> String {
    match bytes.len() {
        0 => "none".to_string(),
        len => format!("0 - {}", len - 1),
    }
}

fn indent(s: String) -> String {
    s.lines().collect::<Vec<_>>().join(&format!("\n{:>14}", ""))
}

#[cfg(test)]
mod tests {
    use super::{Error, redacted, redacted_debug};
    use crate::{
        ast::{Constant, NamedDeBruijn, Term, Type},
        machine::value::Value,
    };
    use std::rc::Rc;

    fn var(index: usize) -> Term<NamedDeBruijn> {
        Term::Var {
            name: NamedDeBruijn {
                text: "x".to_string(),
                index: index.into(),
            }
            .into(),
            uniq_id: 0,
        }
    }

    fn constr(depth: usize) -> Value {
        let mut value = Value::integer(7.into());

        for tag in 0..depth {
            value = Value::Constr {
                tag,
                fields: vec![Value::bool(true), value],
                term_id: 0,
            };
        }

        value
    }

    #[test]
    fn redacted_debug_shows_what_redacting_all_of_it_shows() {
        let one = Term::Constant {
            value: Constant::Integer(1.into()).into(),
            uniq_id: 1,
        };
        let body = Rc::new(var(1).apply(one).delay());

        let values = vec![
            Value::integer(1.into()),
            Value::list(
                Type::Integer,
                vec![
                    Constant::Integer(1.into()).into(),
                    Constant::Integer(2.into()).into(),
                ],
            ),
            Value::Delay {
                body: body.clone(),
                env: Rc::new(vec![]),
                term_id: 3,
            },
            Value::Lambda {
                parameter_name: NamedDeBruijn {
                    text: "y".to_string(),
                    index: 0.into(),
                }
                .into(),
                body,
                env: Rc::new(vec![constr(2), Value::bool(false)]),
                term_id: 4,
            },
            constr(0),
            constr(1),
            constr(3),
            constr(40),
        ];

        let term = var(3).apply(var(2)).force().delay();

        for max_rows in [0, 1, 2, 5, 10, 25, 1000] {
            for value in &values {
                assert_eq!(
                    redacted_debug(value, max_rows),
                    redacted(format!("{value:#?}"), max_rows)
                );
            }

            assert_eq!(
                redacted_debug(&term, max_rows),
                redacted(format!("{term:#?}"), max_rows)
            );
        }
    }

    #[test]
    fn errors_carrying_deep_values_format_in_bounded_space() {
        // All of `{:#?}` would be tens of gigabytes here.
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let error =
                    Error::NonFunctionalApplication(Value::integer(0.into()), constr(100_000));

                let message = error.to_string();

                assert!(message.len() < 1_000, "{message}");
                assert!(
                    message.ends_with("...redacted...\n              }"),
                    "{message}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
