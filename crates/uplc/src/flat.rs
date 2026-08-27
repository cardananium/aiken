use crate::{
    ast::{
        Constant, DeBruijn, FakeNamedDeBruijn, Name, NamedDeBruijn, Program, Term, Type, Unique,
    }, builtins::DefaultFunction, global_uniq::next_uniq_id, machine::runtime::Compressable
};
use num_bigint::BigInt;
use pallas_codec::flat::{
    Flat,
    de::{self, Decode, Decoder},
    en::{self, Encode, Encoder},
};
use pallas_primitives::{Fragment, conway::PlutusData};
use std::{collections::VecDeque, fmt::Debug, rc::Rc};

const BUILTIN_TAG_WIDTH: u32 = 7;
const CONST_TAG_WIDTH: u32 = 4;
const TERM_TAG_WIDTH: u32 = 4;

pub trait Binder<'b>: Encode + Decode<'b> {
    fn binder_encode(&self, e: &mut Encoder) -> Result<(), en::Error>;
    fn binder_decode(d: &mut Decoder) -> Result<Self, de::Error>;
    fn text(&self) -> String;
}

impl<'b, T> Flat<'b> for Program<T> where T: Binder<'b> + Debug {}

impl<'b, T> Program<T>
where
    T: Binder<'b> + Debug,
{
    pub fn from_cbor(bytes: &'b [u8], buffer: &'b mut Vec<u8>) -> Result<Self, de::Error> {
        let mut cbor_decoder = pallas_codec::minicbor::Decoder::new(bytes);

        let flat_bytes = cbor_decoder
            .bytes()
            .map_err(|err| de::Error::Message(err.to_string()))?;

        buffer.extend(flat_bytes);

        Self::unflat(buffer)
    }

    pub fn from_flat(bytes: &'b [u8]) -> Result<Self, de::Error> {
        Self::unflat(bytes)
    }

    pub fn from_hex(
        hex_str: &str,
        cbor_buffer: &'b mut Vec<u8>,
        flat_buffer: &'b mut Vec<u8>,
    ) -> Result<Self, de::Error> {
        let cbor_bytes = hex::decode(hex_str).map_err(|err| de::Error::Message(err.to_string()))?;

        cbor_buffer.extend(cbor_bytes);

        Self::from_cbor(cbor_buffer, flat_buffer)
    }

    /// Convert a program to cbor bytes.
    ///
    /// _note: The cbor bytes of a program are merely
    /// the flat bytes of the program encoded as cbor bytes._
    ///
    /// # Examples
    ///
    /// ```
    /// use uplc::ast::{Program, Name, Term};
    ///
    /// let term = Term::var("x").lambda("x");
    /// let program = Program { version: (1, 0, 0), term };
    ///
    /// assert_eq!(
    ///     program.to_debruijn().unwrap().to_cbor().unwrap(),
    ///     vec![
    ///         0x46, 0x01, 0x00, 0x00,
    ///         0x20, 0x01, 0x01
    ///     ],
    /// );
    /// ```
    pub fn to_cbor(&self) -> Result<Vec<u8>, en::Error> {
        let flat_bytes = self.flat()?;

        let mut bytes = Vec::new();

        let mut cbor_encoder = pallas_codec::minicbor::Encoder::new(&mut bytes);

        cbor_encoder
            .bytes(&flat_bytes)
            .map_err(|err| en::Error::Message(err.to_string()))?;

        Ok(bytes)
    }

    /// Convert a program to a flat bytes.
    ///
    /// _**note**: Convenient so that people don't need to depend on the flat crate
    /// directly to call programs flat function._
    ///
    /// # Examples
    ///
    /// ```
    /// use uplc::ast::{Program, Name, Term};
    ///
    /// let term = Term::var("x").lambda("x");
    /// let program = Program { version: (1, 0, 0), term };
    ///
    /// assert_eq!(
    ///     program
    ///         .to_debruijn()
    ///         .unwrap()
    ///         .to_flat()
    ///         .unwrap(),
    ///     vec![
    ///         0x01, 0x00, 0x00,
    ///         0x20, 0x01, 0x01
    ///     ],
    /// );
    /// ```
    pub fn to_flat(&self) -> Result<Vec<u8>, en::Error> {
        self.flat()
    }

    /// Convert a program to hex encoded cbor bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use uplc::ast::{Program, Name, Term};
    ///
    /// let term = Term::var("x").lambda("x");
    /// let program = Program { version: (1, 0, 0), term };
    ///
    /// assert_eq!(
    ///     program.to_debruijn().unwrap().to_hex().unwrap(),
    ///     "46010000200101".to_string(),
    /// );
    /// ```
    pub fn to_hex(&self) -> Result<String, en::Error> {
        let bytes = self.to_cbor()?;

        let hex = hex::encode(bytes);

        Ok(hex)
    }
}

impl<'b, T> Encode for Program<T>
where
    T: Binder<'b> + Debug,
{
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        let (major, minor, patch) = self.version;

        major.encode(e)?;
        minor.encode(e)?;
        patch.encode(e)?;

        self.term.encode(e)?;

        Ok(())
    }
}

impl<'b, T> Decode<'b> for Program<T>
where
    T: Binder<'b>,
{
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        let version = (usize::decode(d)?, usize::decode(d)?, usize::decode(d)?);
        // Flat is bit-addressed, so a position is a byte offset AND a bit
        // offset within that byte. Rewinding needs both: restoring `pos`
        // alone resumes on the right byte at whatever bit the failed decode
        // stopped on.
        let (term_pos, term_used_bits) = (d.pos, d.used_bits);

        match Term::decode(d) {
            Ok(term) => Ok(Program { version, term }),
            // The plain decoder reports what went wrong but not where in the
            // term it was. Rewind and re-run the same walk with a state log,
            // which names the path it took, purely to build the message.
            Err(fast_error) => {
                d.pos = term_pos;
                d.used_bits = term_used_bits;

                let mut state_log: Vec<String> = vec![];

                match Term::<T>::decode_debug(d, &mut state_log) {
                    Ok(_) => Err(fast_error),
                    Err(error) => Err(de::Error::Message(format!(
                        "{} {error}",
                        state_log.join("")
                    ))),
                }
            }
        }
    }
}

/// Emit one element of a flat list, or close the list.
///
/// `Encoder::encode_list_with` writes a `1` bit before every element and a `0` to close, but it
/// takes the element encoder as a function, which is exactly the recursion we are trying to avoid.
/// These two steps reproduce its bit pattern from inside the iterative walk instead. `Encoder::one`
/// and `Encoder::zero` are private to `pallas-codec`; `Encoder::bool` is the public spelling and
/// calls straight through to them, so the bits are identical.
enum ListStep {
    Item,
    End,
}

/// Read the continuation bit that precedes each element of a flat list.
///
/// `Decoder::decode_list_with` does this with `Decoder::bit`, but it takes the element decoder as a
/// function, which is exactly the recursion we are trying to avoid. `bit` is private to
/// `pallas-codec`, and `Decoder::bool` -- the public one-bit read -- skips the bounds check and
/// would panic on a truncated script rather than fail. `bits8(1)` reads the same bit and advances
/// identically, and the guard in front of it reproduces `bit`'s `EndOfBuffer` instead of `bits8`'s
/// `NotEnoughBits`, so a malformed script still fails exactly the way it always did.
fn decode_list_bit(d: &mut Decoder) -> Result<bool, de::Error> {
    if d.pos >= d.buffer.len() {
        return Err(de::Error::EndOfBuffer);
    }

    Ok(d.bits8(1)? != 0)
}

impl<'b, T> Encode for Term<T>
where
    T: Binder<'b> + Debug,
{
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        // Encoding walks the same spine `decode` does, so it has to be iterative for the same
        // reason: the nesting depth of a term is attacker-controlled -- the cheapest level in flat
        // is a 4-bit `delay` tag, half a byte, so a script inside the on-chain size limit is tens
        // of thousands of levels deep -- and on `wasm32` the engine's call stack is about a
        // megabyte and cannot be grown from the page. Recursing here capped the whole
        // decode-then-re-encode round-trip at roughly 10,500 levels, measured. Do not fold this
        // back into a recursive walk.
        //
        // `Step` is the encoder's mirror of the `Frame` enum in `Term::decode`: the stack holds
        // what is left to emit. It is LIFO, so children go on in reverse of the order they appear
        // in the output.
        enum Step<'a, U> {
            /// Emit this term: its tag, its immediate payload, then its children.
            Term(&'a Term<U>),
            /// Emit a list's continuation bit before an element, or its terminating bit.
            List(ListStep),
        }

        let mut steps: Vec<Step<'_, T>> = vec![Step::Term(self)];

        while let Some(step) = steps.pop() {
            let term = match step {
                Step::List(ListStep::Item) => {
                    e.bool(true);
                    continue;
                }
                Step::List(ListStep::End) => {
                    e.bool(false);
                    continue;
                }
                Step::Term(term) => term,
            };

            match term {
                Term::Var { name, .. } => {
                    encode_term_tag(0, e)?;
                    name.encode(e)?;
                }
                Term::Delay { body, .. } => {
                    encode_term_tag(1, e)?;
                    steps.push(Step::Term(body.as_ref()));
                }
                Term::Lambda {
                    parameter_name,
                    body,
                    ..
                } => {
                    encode_term_tag(2, e)?;
                    parameter_name.binder_encode(e)?;
                    steps.push(Step::Term(body.as_ref()));
                }
                Term::Apply { function, argument, .. } => {
                    encode_term_tag(3, e)?;
                    steps.push(Step::Term(argument.as_ref()));
                    steps.push(Step::Term(function.as_ref()));
                }

                Term::Constant { value: constant, .. } => {
                    encode_term_tag(4, e)?;
                    constant.encode(e)?;
                }

                Term::Force { body, .. } => {
                    encode_term_tag(5, e)?;
                    steps.push(Step::Term(body.as_ref()));
                }

                Term::Error { .. } => {
                    encode_term_tag(6, e)?;
                }
                Term::Builtin { fun, .. } => {
                    encode_term_tag(7, e)?;

                    fun.encode(e)?;
                }
                Term::Constr { tag, fields, .. } => {
                    encode_term_tag(8, e)?;

                    tag.encode(e)?;

                    steps.push(Step::List(ListStep::End));

                    for field in fields.iter().rev() {
                        steps.push(Step::Term(field));
                        steps.push(Step::List(ListStep::Item));
                    }
                }
                Term::Case { constr, branches, .. } => {
                    encode_term_tag(9, e)?;

                    steps.push(Step::List(ListStep::End));

                    for branch in branches.iter().rev() {
                        steps.push(Step::Term(branch));
                        steps.push(Step::List(ListStep::Item));
                    }

                    steps.push(Step::Term(constr.as_ref()));
                }
            }
        }

        Ok(())
    }
}

/// Record one step of the path into the term, when the caller asked for one.
///
/// The step is only built inside the `if`, so the plain decoder pays nothing for the bookkeeping
/// that only the error reporter needs.
///
/// Pass interpolated values as explicit arguments -- `note!(log, "{})", fun)`, not
/// `note!(log, "{fun})")`. A format string that captures its values implicitly has no arguments,
/// so it matches the first arm below and gets recorded with the braces still in it.
macro_rules! note {
    // A step with nothing interpolated into it.
    ($log:expr, $step:literal) => {
        if let Some(log) = $log.as_deref_mut() {
            log.push(String::from($step));
        }
    };
    // A step built from values.
    ($log:expr, $fmt:literal, $($arg:tt)+) => {
        if let Some(log) = $log.as_deref_mut() {
            log.push(format!($fmt, $($arg)+));
        }
    };
}

/// A constructor the term walk has opened and not yet finished, while it reads the next child.
/// The encoder's mirror of this is the `Step` enum in `Term::encode`.
enum Frame<U> {
    Delay,
    Force,
    Lambda { parameter_name: Rc<U> },
    ApplyFn,
    ApplyArg { function: Rc<Term<U>> },
    /// Fields read so far for a `constr`; one more is on its way.
    ConstrField { tag: usize, fields: Vec<Term<U>> },
    /// The scrutinee of a `case` is being read.
    CaseConstr,
    /// Branches read so far for a `case`; one more is on its way.
    CaseBranch { constr: Rc<Term<U>>, branches: Vec<Term<U>> },
}

impl<'b, T> Decode<'b> for Term<T>
where
    T: Binder<'b>,
{
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Term::decode_with_log(d, None)
    }
}

impl<'b, T> Term<T>
where
    T: Binder<'b>,
{
    /// Decode a term, recording the path into it for an error message.
    fn decode_debug(d: &mut Decoder, state_log: &mut Vec<String>) -> Result<Term<T>, de::Error> {
        Term::decode_with_log(d, Some(state_log))
    }

    /// The one term walk, shared by `Term::decode` and `Term::decode_debug`.
    ///
    /// It is iterative because the nesting depth of a term is attacker-controlled: the cheapest
    /// level in flat is a 4-bit `delay` tag, half a byte, so a script inside the on-chain size
    /// limit is tens of thousands of levels deep, and on `wasm32` the engine's call stack is about
    /// a megabyte and cannot be grown from the page. `frames` holds what the walk has opened and
    /// not yet finished, in place of the call stack a recursive descent would use. Do not fold
    /// this back into a recursive walk.
    ///
    /// `state_log` is the only thing separating the two entry points: when it is `Some`, the walk
    /// names each constructor it enters so a failure can be reported as a position inside the
    /// term rather than just a reason. Sharing one body is what keeps the two from drifting: the
    /// plain decoder cannot start accepting something the reporter would reject.
    fn decode_with_log(
        d: &mut Decoder,
        mut state_log: Option<&mut Vec<String>>,
    ) -> Result<Term<T>, de::Error> {
        let mut frames: Vec<Frame<T>> = Vec::new();

        let result = Term::decode_frames(d, state_log.as_deref_mut(), &mut frames);

        // On the way out of a failure, close every constructor still open, innermost first. The
        // recursive decoder used to do this as the error travelled back up through its callers;
        // here the frames left standing are exactly those callers.
        if result.is_err()
            && let Some(log) = state_log
        {
            for frame in frames.iter().rev() {
                match frame {
                    Frame::Delay | Frame::Force | Frame::Lambda { .. } => {
                        log.push(")".to_string())
                    }
                    Frame::ApplyFn => log.push(" not parsed]".to_string()),
                    Frame::ApplyArg { .. } => log.push("]".to_string()),
                    Frame::ConstrField { .. } | Frame::CaseConstr | Frame::CaseBranch { .. } => {}
                }
            }
        }

        result
    }

    /// Walk the term, leaving whatever is still open in `frames` if it fails.
    fn decode_frames(
        d: &mut Decoder,
        mut state_log: Option<&mut Vec<String>>,
        frames: &mut Vec<Frame<T>>,
    ) -> Result<Term<T>, de::Error> {
        let mut current: Option<Term<T>> = None;

        loop {
            if current.is_none() {
                // Read one constructor. The ones with children push a frame and go round again
                // for the first of them; the rest are complete and start the fold below.
                let parsed = match decode_term_tag(d)? {
                    0 => {
                        note!(state_log, "(var ");

                        match T::decode(d) {
                            Ok(name) => {
                                note!(state_log, "{})", name.text());

                                Term::Var {
                                    name: name.into(),
                                    uniq_id: next_uniq_id(),
                                }
                            }
                            Err(error) => {
                                note!(state_log, "parse error)");

                                return Err(error);
                            }
                        }
                    }
                    1 => {
                        note!(state_log, "(delay ");
                        frames.push(Frame::Delay);
                        continue;
                    }
                    2 => {
                        note!(state_log, "(lam ");

                        match T::binder_decode(d) {
                            Ok(parameter_name) => {
                                note!(state_log, "{}", parameter_name.text());

                                frames.push(Frame::Lambda {
                                    parameter_name: parameter_name.into(),
                                });

                                continue;
                            }
                            Err(error) => {
                                note!(state_log, ")");

                                return Err(error);
                            }
                        }
                    }
                    3 => {
                        note!(state_log, "[ ");
                        frames.push(Frame::ApplyFn);
                        continue;
                    }
                    // Need size limit for Constant
                    4 => {
                        note!(state_log, "(con ");

                        match Constant::decode(d) {
                            Ok(constant) => {
                                note!(state_log, "{})", constant.to_pretty());

                                Term::Constant {
                                    value: constant.into(),
                                    uniq_id: next_uniq_id(),
                                }
                            }
                            Err(error) => {
                                note!(state_log, "parse error)");

                                return Err(error);
                            }
                        }
                    }
                    5 => {
                        note!(state_log, "(force ");
                        frames.push(Frame::Force);
                        continue;
                    }
                    6 => {
                        note!(state_log, "(error)");

                        Term::Error {
                            uniq_id: next_uniq_id(),
                        }
                    }
                    7 => {
                        note!(state_log, "(builtin ");

                        match DefaultFunction::decode(d) {
                            Ok(fun) => {
                                note!(state_log, "{})", fun);

                                Term::Builtin {
                                    fun,
                                    uniq_id: next_uniq_id(),
                                }
                            }
                            Err(error) => {
                                note!(state_log, "parse error)");

                                return Err(error);
                            }
                        }
                    }
                    8 => {
                        note!(state_log, "(constr ");

                        let tag = usize::decode(d)?;

                        // The fields are walked a frame at a time rather than with
                        // `decode_list_with`, whose element decoder would be this function again.
                        if decode_list_bit(d)? {
                            frames.push(Frame::ConstrField {
                                tag,
                                fields: Vec::new(),
                            });

                            continue;
                        }

                        Term::Constr {
                            tag,
                            fields: Vec::new(),
                            uniq_id: next_uniq_id(),
                        }
                    }
                    9 => {
                        note!(state_log, "(case ");
                        frames.push(Frame::CaseConstr);
                        continue;
                    }
                    x => {
                        note!(state_log, "parse error");

                        let buffer_slice: Vec<u8> = d
                            .buffer
                            .to_vec()
                            .iter()
                            .skip(d.pos.saturating_sub(5))
                            .take(10)
                            .cloned()
                            .collect();

                        return Err(de::Error::UnknownTermConstructor(
                            x,
                            if d.pos > 5 { 5 } else { d.pos },
                            format!("{buffer_slice:02X?}"),
                            d.pos,
                            d.buffer.len(),
                        ));
                    }
                };

                current = Some(parsed);
            }

            // Fold the finished term into whatever was waiting for it, until something needs
            // another child read first.
            while let Some(frame) = frames.pop() {
                match frame {
                    Frame::Delay => {
                        let body = Rc::new(current.take().expect("term present"));

                        note!(state_log, ")");

                        current = Some(Term::Delay {
                            body,
                            uniq_id: next_uniq_id(),
                        });
                    }
                    Frame::Force => {
                        let body = Rc::new(current.take().expect("term present"));

                        note!(state_log, ")");

                        current = Some(Term::Force {
                            body,
                            uniq_id: next_uniq_id(),
                        });
                    }
                    Frame::Lambda { parameter_name } => {
                        let body = Rc::new(current.take().expect("term present"));

                        note!(state_log, ")");

                        current = Some(Term::Lambda {
                            parameter_name,
                            body,
                            uniq_id: next_uniq_id(),
                        });
                    }
                    Frame::ApplyFn => {
                        let function = Rc::new(current.take().expect("term present"));

                        note!(state_log, " ");

                        frames.push(Frame::ApplyArg { function });
                        current = None;
                        break;
                    }
                    Frame::ApplyArg { function } => {
                        let argument = Rc::new(current.take().expect("term present"));

                        note!(state_log, "]");

                        current = Some(Term::Apply {
                            function,
                            argument,
                            uniq_id: next_uniq_id(),
                        });
                    }
                    Frame::ConstrField { tag, mut fields } => {
                        fields.push(current.take().expect("term present"));

                        if decode_list_bit(d)? {
                            frames.push(Frame::ConstrField { tag, fields });
                            current = None;
                            break;
                        }

                        current = Some(Term::Constr {
                            tag,
                            fields,
                            uniq_id: next_uniq_id(),
                        });
                    }
                    Frame::CaseConstr => {
                        let constr = Rc::new(current.take().expect("term present"));

                        if decode_list_bit(d)? {
                            frames.push(Frame::CaseBranch {
                                constr,
                                branches: Vec::new(),
                            });
                            current = None;
                            break;
                        }

                        current = Some(Term::Case {
                            constr,
                            branches: Vec::new(),
                            uniq_id: next_uniq_id(),
                        });
                    }
                    Frame::CaseBranch {
                        constr,
                        mut branches,
                    } => {
                        branches.push(current.take().expect("term present"));

                        if decode_list_bit(d)? {
                            frames.push(Frame::CaseBranch { constr, branches });
                            current = None;
                            break;
                        }

                        current = Some(Term::Case {
                            constr,
                            branches,
                            uniq_id: next_uniq_id(),
                        });
                    }
                }
            }

            if frames.is_empty()
                && let Some(done) = current.take()
            {
                return Ok(done);
            }
        }
    }
}

/// Integers are typically smaller so we save space
/// by encoding them in 7 bits and this allows it to be byte alignment agnostic.
/// Strings and bytestrings span multiple bytes so using bytestring is
/// the most effective encoding.
/// i.e. A 17 or greater length byte array loses efficiency being encoded as
/// a unsigned integer instead of a byte array
impl Encode for Constant {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        match self {
            Constant::Integer(i) => {
                encode_constant(&[0], e)?;
                i.encode(e)?;
            }

            Constant::ByteString(bytes) => {
                encode_constant(&[1], e)?;
                bytes.encode(e)?;
            }
            Constant::String(s) => {
                encode_constant(&[2], e)?;
                s.encode(e)?;
            }
            Constant::Unit => encode_constant(&[3], e)?,
            Constant::Bool(b) => {
                encode_constant(&[4], e)?;
                b.encode(e)?;
            }
            Constant::ProtoList(typ, list) => {
                let mut type_encode = vec![7, 5];

                encode_type(typ, &mut type_encode);

                encode_constant(&type_encode, e)?;

                e.encode_list_with(list, encode_constant_value)?;
            }
            Constant::ProtoPair(type1, type2, a, b) => {
                let mut type_encode = vec![7, 7, 6];

                encode_type(type1, &mut type_encode);

                encode_type(type2, &mut type_encode);

                encode_constant(&type_encode, e)?;
                encode_constant_value(a, e)?;
                encode_constant_value(b, e)?;
            }
            Constant::Data(data) => {
                encode_constant(&[8], e)?;

                let cbor = data
                    .encode_fragment()
                    .map_err(|err| en::Error::Message(err.to_string()))?;

                cbor.encode(e)?;
            }
            Constant::Bls12_381G1Element(_) => {
                encode_constant(&[9], e)?;

                return Err(en::Error::Message(
                    "BLS12-381 G1 points are not supported for flat encoding".to_string(),
                ));
            }
            Constant::Bls12_381G2Element(_) => {
                encode_constant(&[10], e)?;

                return Err(en::Error::Message(
                    "BLS12-381 G2 points are not supported for flat encoding".to_string(),
                ));
            }
            Constant::Bls12_381MlResult(_) => {
                encode_constant(&[11], e)?;

                return Err(en::Error::Message(
                    "BLS12-381 ML results are not supported for flat encoding".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Write a constant's payload, without the type tags that introduce it.
///
/// Iterative for the same reason the term coders are: `ProtoList` and `ProtoPair` nest, and the
/// depth is attacker-controlled. `Step` is the same device `Term::encode` uses -- the stack holds
/// what is left to emit, and because it is LIFO, children go on in reverse of the order they
/// appear in the output.
fn encode_constant_value(x: &Constant, e: &mut Encoder) -> Result<(), en::Error> {
    enum Step<'a> {
        Value(&'a Constant),
        List(ListStep),
    }

    let mut steps: Vec<Step<'_>> = vec![Step::Value(x)];

    while let Some(step) = steps.pop() {
        let constant = match step {
            Step::List(ListStep::Item) => {
                e.bool(true);
                continue;
            }
            Step::List(ListStep::End) => {
                e.bool(false);
                continue;
            }
            Step::Value(constant) => constant,
        };

        match constant {
            Constant::Integer(x) => x.encode(e)?,
            Constant::ByteString(b) => b.encode(e)?,
            Constant::String(s) => s.encode(e)?,
            Constant::Unit => (),
            Constant::Bool(b) => b.encode(e)?,
            Constant::ProtoList(_, list) => {
                steps.push(Step::List(ListStep::End));

                for item in list.iter().rev() {
                    steps.push(Step::Value(item));
                    steps.push(Step::List(ListStep::Item));
                }
            }
            Constant::ProtoPair(_, _, a, b) => {
                steps.push(Step::Value(b.as_ref()));
                steps.push(Step::Value(a.as_ref()));
            }
            Constant::Data(data) => {
                let cbor = data
                    .encode_fragment()
                    .map_err(|err| en::Error::Message(err.to_string()))?;

                cbor.encode(e)?
            }
            Constant::Bls12_381G1Element(_) => {
                return Err(en::Error::Message(
                    "BLS12-381 G1 points are not supported for flat encoding".to_string(),
                ));
            }
            Constant::Bls12_381G2Element(_) => {
                return Err(en::Error::Message(
                    "BLS12-381 G2 points are not supported for flat encoding".to_string(),
                ));
            }
            Constant::Bls12_381MlResult(_) => {
                return Err(en::Error::Message(
                    "BLS12-381 ML results are not supported for flat encoding".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Flatten a type into its run of 4-bit constructor tags.
///
/// Iterative for the same reason the term coders are: a nested `list` costs ten bits of tag per
/// level, so a script inside the on-chain size limit carries a type over ten thousand levels deep,
/// and that depth is attacker-controlled. The stack is LIFO, so a pair's components go on in
/// reverse of the order they are written.
fn encode_type(typ: &Type, bytes: &mut Vec<u8>) {
    let mut pending: Vec<&Type> = vec![typ];

    while let Some(typ) = pending.pop() {
        match typ {
            Type::Integer => bytes.push(0),
            Type::ByteString => bytes.push(1),
            Type::String => bytes.push(2),
            Type::Unit => bytes.push(3),
            Type::Bool => bytes.push(4),
            Type::List(sub_typ) => {
                bytes.extend(vec![7, 5]);
                pending.push(sub_typ);
            }
            Type::Pair(type1, type2) => {
                bytes.extend(vec![7, 7, 6]);
                pending.push(type2);
                pending.push(type1);
            }
            Type::Data => bytes.push(8),
            Type::Bls12_381G1Element => bytes.push(9),
            Type::Bls12_381G2Element => bytes.push(10),
            Type::Bls12_381MlResult => bytes.push(11),
        }
    }
}

impl Decode<'_> for Constant {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        match &decode_constant(d)?[..] {
            [0] => Ok(Constant::Integer(BigInt::decode(d)?)),
            [1] => Ok(Constant::ByteString(Vec::<u8>::decode(d)?)),
            [2] => Ok(Constant::String(String::decode(d)?)),
            [3] => Ok(Constant::Unit),
            [4] => Ok(Constant::Bool(bool::decode(d)?)),
            [7, 5, rest @ ..] => {
                let mut rest = VecDeque::from(rest.to_vec());

                let typ = decode_type(&mut rest)?;

                let list: Vec<Constant> =
                    d.decode_list_with(|d| decode_constant_value(typ.clone().into(), d))?;

                Ok(Constant::ProtoList(typ, list))
            }
            [7, 7, 6, rest @ ..] => {
                let mut rest = VecDeque::from(rest.to_vec());

                let type1 = decode_type(&mut rest)?;
                let type2 = decode_type(&mut rest)?;

                let a = decode_constant_value(type1.clone().into(), d)?;
                let b = decode_constant_value(type2.clone().into(), d)?;

                Ok(Constant::ProtoPair(type1, type2, a.into(), b.into()))
            }
            [8] => {
                let cbor = Vec::<u8>::decode(d)?;

                let data = PlutusData::decode_fragment(&cbor)
                    .map_err(|err| de::Error::Message(err.to_string()))?;

                Ok(Constant::Data(data))
            }
            [9] => {
                let p1 = Vec::<u8>::decode(d)?;

                let _p1 = blst::blst_p1::uncompress(&p1)
                    .map_err(|err| de::Error::Message(format!("Failed to uncompress p1: {err}")))?;

                Err(de::Error::Message(
                    "BLS12-381 G1 points are not supported for flat decoding.".to_string(),
                ))
            }

            [10] => {
                let p2 = Vec::<u8>::decode(d)?;

                let _p2 = blst::blst_p2::uncompress(&p2)
                    .map_err(|err| de::Error::Message(format!("Failed to uncompress p2: {err}")))?;

                Err(de::Error::Message(
                    "BLS12-381 G2 points are not supported for flat decoding.".to_string(),
                ))
            }
            [11] => Err(de::Error::Message(
                "BLS12-381 ML results are not supported for flat decoding".to_string(),
            )),
            x => Err(de::Error::Message(format!(
                "Unknown constant constructor tag: {x:?}"
            ))),
        }
    }
}

/// Read a constant's payload, given the type that introduces it.
///
/// The mirror of `encode_constant_value`, and iterative for the same reason: `list` and `pair`
/// nest as deep as the script says. `Frame` is the same device `Term::decode` uses -- it records
/// what is still waiting for a value while the walk reads the next one.
fn decode_constant_value(typ: Rc<Type>, d: &mut Decoder) -> Result<Constant, de::Error> {
    enum Frame {
        /// Elements of a `list` read so far; another one is on its way.
        ListItem {
            elem_type: Rc<Type>,
            items: Vec<Constant>,
        },
        /// The first component of a `pair` is being read.
        PairFst {
            first_type: Rc<Type>,
            second_type: Rc<Type>,
        },
        /// The second component of a `pair` is being read; the first is done.
        PairSnd {
            first_type: Rc<Type>,
            second_type: Rc<Type>,
            first: Rc<Constant>,
        },
    }

    let mut frames: Vec<Frame> = Vec::new();
    let mut next = typ;

    loop {
        let typ = Rc::clone(&next);

        // Read one value. `list` and `pair` push a frame and go round again for their first
        // component; everything else is a leaf and starts the fold below.
        let mut current = match typ.as_ref() {
            Type::Integer => Constant::Integer(BigInt::decode(d)?),
            Type::ByteString => Constant::ByteString(Vec::<u8>::decode(d)?),
            Type::String => Constant::String(String::decode(d)?),
            Type::Unit => Constant::Unit,
            Type::Bool => Constant::Bool(bool::decode(d)?),
            Type::List(sub_type) => {
                // Walked a frame at a time rather than with `decode_list_with`, whose element
                // decoder would be this function again.
                if decode_list_bit(d)? {
                    next = Rc::clone(sub_type);

                    frames.push(Frame::ListItem {
                        elem_type: Rc::clone(sub_type),
                        items: Vec::new(),
                    });

                    continue;
                }

                Constant::ProtoList(sub_type.as_ref().clone(), Vec::new())
            }
            Type::Pair(type1, type2) => {
                next = Rc::clone(type1);

                frames.push(Frame::PairFst {
                    first_type: Rc::clone(type1),
                    second_type: Rc::clone(type2),
                });

                continue;
            }
            Type::Data => {
                let cbor = Vec::<u8>::decode(d)?;

                let data = PlutusData::decode_fragment(&cbor)
                    .map_err(|err| de::Error::Message(err.to_string()))?;

                Constant::Data(data)
            }
            Type::Bls12_381G1Element => {
                let p1 = Vec::<u8>::decode(d)?;

                let _p1 = blst::blst_p1::uncompress(&p1)
                    .map_err(|err| de::Error::Message(format!("Failed to uncompress p1: {err}")))?;

                return Err(de::Error::Message(
                    "BLS12-381 G1 points are not supported for flat decoding.".to_string(),
                ));
            }
            Type::Bls12_381G2Element => {
                let p2 = Vec::<u8>::decode(d)?;

                let _p2 = blst::blst_p2::uncompress(&p2)
                    .map_err(|err| de::Error::Message(format!("Failed to uncompress p2: {err}")))?;

                return Err(de::Error::Message(
                    "BLS12-381 G2 points are not supported for flat decoding.".to_string(),
                ));
            }
            Type::Bls12_381MlResult => {
                return Err(de::Error::Message(
                    "BLS12-381 ML results are not supported for flat decoding".to_string(),
                ));
            }
        };

        // Fold the finished value into whatever was waiting for it.
        loop {
            match frames.pop() {
                None => return Ok(current),
                Some(Frame::ListItem {
                    elem_type,
                    mut items,
                }) => {
                    items.push(current);

                    if decode_list_bit(d)? {
                        next = Rc::clone(&elem_type);
                        frames.push(Frame::ListItem { elem_type, items });
                        break;
                    }

                    current = Constant::ProtoList(elem_type.as_ref().clone(), items);
                }
                Some(Frame::PairFst {
                    first_type,
                    second_type,
                }) => {
                    next = Rc::clone(&second_type);

                    frames.push(Frame::PairSnd {
                        first_type,
                        second_type,
                        first: current.into(),
                    });

                    break;
                }
                Some(Frame::PairSnd {
                    first_type,
                    second_type,
                    first,
                }) => {
                    current = Constant::ProtoPair(
                        first_type.as_ref().clone(),
                        second_type.as_ref().clone(),
                        first,
                        current.into(),
                    );
                }
            }
        }
    }
}

/// Rebuild a type from its run of 4-bit constructor tags.
///
/// The mirror of `encode_type`, and iterative for the same reason: the depth of the type is
/// attacker-controlled. `Frame` plays the same role here that it does in `Term::decode` -- it
/// records what is still waiting for an operand while the walk reads the next one.
fn decode_type(types: &mut VecDeque<u8>) -> Result<Type, de::Error> {
    enum Frame {
        /// The element type of a `list` is being read.
        List,
        /// The first component of a `pair` is being read.
        PairFst,
        /// The second component of a `pair` is being read; the first is done.
        PairSnd { first: Rc<Type> },
    }

    let mut frames: Vec<Frame> = Vec::new();

    loop {
        // Read one constructor. `list` and `pair` push a frame and go round again for their
        // operand; everything else is a leaf and starts the fold below.
        let mut current = match types.pop_front() {
            Some(4) => Type::Bool,
            Some(0) => Type::Integer,
            Some(2) => Type::String,
            Some(1) => Type::ByteString,
            Some(3) => Type::Unit,
            Some(8) => Type::Data,
            Some(9) => Type::Bls12_381G1Element,
            Some(10) => Type::Bls12_381G2Element,
            Some(11) => Type::Bls12_381MlResult,
            Some(7) => match types.pop_front() {
                Some(5) => {
                    frames.push(Frame::List);
                    continue;
                }
                Some(7) => match types.pop_front() {
                    Some(6) => {
                        frames.push(Frame::PairFst);
                        continue;
                    }
                    Some(x) => {
                        return Err(de::Error::Message(format!(
                            "Unknown constant type tag: {x}"
                        )));
                    }
                    None => {
                        return Err(de::Error::Message("Unexpected empty buffer".to_string()));
                    }
                },
                Some(x) => {
                    return Err(de::Error::Message(format!(
                        "Unknown constant type tag: {x}"
                    )));
                }
                None => {
                    return Err(de::Error::Message("Unexpected empty buffer".to_string()));
                }
            },

            Some(x) => {
                return Err(de::Error::Message(format!(
                    "Unknown constant type tag: {x}"
                )));
            }
            None => {
                return Err(de::Error::Message("Unexpected empty buffer".to_string()));
            }
        };

        // Fold the finished type into whatever was waiting for it.
        loop {
            match frames.pop() {
                None => return Ok(current),
                Some(Frame::List) => current = Type::List(current.into()),
                Some(Frame::PairFst) => {
                    frames.push(Frame::PairSnd {
                        first: current.into(),
                    });
                    break;
                }
                Some(Frame::PairSnd { first }) => current = Type::Pair(first, current.into()),
            }
        }
    }
}

impl Encode for Unique {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        isize::from(*self).encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for Unique {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(isize::decode(d)?.into())
    }
}

impl Encode for Name {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.text.encode(e)?;
        self.unique.encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for Name {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(Name {
            text: String::decode(d)?,
            unique: Unique::decode(d)?,
        })
    }
}

impl Binder<'_> for Name {
    fn binder_encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.encode(e)?;

        Ok(())
    }

    fn binder_decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Name::decode(d)
    }

    fn text(&self) -> String {
        self.text.clone()
    }
}

impl Encode for NamedDeBruijn {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.text.encode(e)?;
        self.index.encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for NamedDeBruijn {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(NamedDeBruijn {
            text: String::decode(d)?,
            index: DeBruijn::decode(d)?,
        })
    }
}

impl Binder<'_> for NamedDeBruijn {
    fn binder_encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.text.encode(e)?;
        self.index.encode(e)?;

        Ok(())
    }

    fn binder_decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(NamedDeBruijn {
            text: String::decode(d)?,
            index: DeBruijn::decode(d)?,
        })
    }

    fn text(&self) -> String {
        format!("{}_{}", &self.text, self.index)
    }
}

impl Encode for DeBruijn {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        usize::from(*self).encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for DeBruijn {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(usize::decode(d)?.into())
    }
}

impl Binder<'_> for DeBruijn {
    fn binder_encode(&self, _: &mut Encoder) -> Result<(), en::Error> {
        Ok(())
    }

    fn binder_decode(_d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(DeBruijn::new(0))
    }

    fn text(&self) -> String {
        format!("i_{self}")
    }
}

impl Encode for FakeNamedDeBruijn {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        let index: DeBruijn = self.clone().into();

        index.encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for FakeNamedDeBruijn {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        let index = DeBruijn::decode(d)?;

        Ok(index.into())
    }
}

impl Binder<'_> for FakeNamedDeBruijn {
    fn binder_encode(&self, _: &mut Encoder) -> Result<(), en::Error> {
        Ok(())
    }

    fn binder_decode(_d: &mut Decoder) -> Result<Self, de::Error> {
        let index = DeBruijn::new(0);

        Ok(index.into())
    }

    fn text(&self) -> String {
        format!("{}_{}", self.0.text, self.0.index)
    }
}

impl Encode for DefaultFunction {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        e.bits(BUILTIN_TAG_WIDTH as i64, *self as u8);

        Ok(())
    }
}

impl Decode<'_> for DefaultFunction {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        let builtin_tag = d.bits8(BUILTIN_TAG_WIDTH as usize)?;
        builtin_tag.try_into()
    }
}

fn encode_term_tag(tag: u8, e: &mut Encoder) -> Result<(), en::Error> {
    safe_encode_bits(TERM_TAG_WIDTH, tag, e)
}

fn decode_term_tag(d: &mut Decoder) -> Result<u8, de::Error> {
    d.bits8(TERM_TAG_WIDTH as usize)
}

fn safe_encode_bits(num_bits: u32, byte: u8, e: &mut Encoder) -> Result<(), en::Error> {
    if 2_u8.pow(num_bits) <= byte {
        Err(en::Error::Message(format!(
            "Overflow detected, cannot fit {byte} in {num_bits} bits."
        )))
    } else {
        e.bits(num_bits as i64, byte);
        Ok(())
    }
}

pub fn encode_constant(tag: &[u8], e: &mut Encoder) -> Result<(), en::Error> {
    e.encode_list_with(tag, encode_constant_tag)?;

    Ok(())
}

pub fn decode_constant(d: &mut Decoder) -> Result<Vec<u8>, de::Error> {
    d.decode_list_with(decode_constant_tag)
}

pub fn encode_constant_tag(tag: &u8, e: &mut Encoder) -> Result<(), en::Error> {
    safe_encode_bits(CONST_TAG_WIDTH, *tag, e)
}

pub fn decode_constant_tag(d: &mut Decoder) -> Result<u8, de::Error> {
    d.bits8(CONST_TAG_WIDTH as usize)
}

#[cfg(test)]
mod tests {
    use super::{Constant, Program, Term};
    use crate::{
        ast::{DeBruijn, Name, Type}, global_uniq::next_uniq_id, parser
    };
    use indoc::indoc;
    use pallas_codec::flat::Flat;

    #[test]
    fn flat_encode_integer() {
        let program = Program::<Name> {
            version: (11, 22, 33),
            term: Term::Constant {
                value: Constant::Integer(11.into()).into(),
                uniq_id: next_uniq_id(),
            },
        };

        let expected_bytes = vec![
            0b00001011, 0b00010110, 0b00100001, 0b01001000, 0b00000101, 0b10000001,
        ];

        let actual_bytes = program.to_flat().unwrap();

        assert_eq!(actual_bytes, expected_bytes)
    }

    #[test]
    fn flat_encode_list_list_integer() {
        let program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant{
                value: Constant::ProtoList(
                    Type::List(Type::Integer.into()),
                    vec![
                        Constant::ProtoList(Type::Integer, vec![Constant::Integer(7.into())]),
                        Constant::ProtoList(Type::Integer, vec![Constant::Integer(5.into())]),
                    ],
                )
                .into(),
                uniq_id: next_uniq_id(),
            },
        };

        let expected_bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11010110, 0b11110101, 0b10000011,
            0b00001110, 0b01100001, 0b01000001,
        ];

        let actual_bytes = program.to_flat().unwrap();

        assert_eq!(actual_bytes, expected_bytes)
    }

    #[test]
    fn flat_encode_pair_pair_integer_bool_integer() {
        let program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant {
                value: Constant::ProtoPair(
                    Type::Pair(Type::Integer.into(), Type::Bool.into()),
                    Type::Integer,
                    Constant::ProtoPair(
                        Type::Integer,
                        Type::Bool,
                        Constant::Integer(11.into()).into(),
                        Constant::Bool(true).into(),
                    )
                    .into(),
                    Constant::Integer(11.into()).into(),
                )
                .into(),
                uniq_id: next_uniq_id(),
            },
        };

        let expected_bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11011110, 0b11010111, 0b10111101,
            0b10100001, 0b01001000, 0b00000101, 0b10100010, 0b11000001,
        ];

        let actual_bytes = program.to_flat().unwrap();

        assert_eq!(actual_bytes, expected_bytes)
    }

    #[test]
    fn flat_decode_list_list_integer() {
        let bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11010110, 0b11110101, 0b10000011,
            0b00001110, 0b01100001, 0b01000001,
        ];

        let expected_program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant {
                value: Constant::ProtoList(
                    Type::List(Type::Integer.into()),
                    vec![
                        Constant::ProtoList(Type::Integer, vec![Constant::Integer(7.into())]),
                        Constant::ProtoList(Type::Integer, vec![Constant::Integer(5.into())]),
                    ],
                )
                .into(),
                uniq_id: next_uniq_id(),
            },
        };

        let actual_program: Program<Name> = Program::unflat(&bytes).unwrap();

        assert_eq!(actual_program, expected_program)
    }

    #[test]
    fn flat_decode_pair_pair_integer_bool_integer() {
        let bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11011110, 0b11010111, 0b10111101,
            0b10100001, 0b01001000, 0b00000101, 0b10100010, 0b11000001,
        ];

        let expected_program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant {
                value: Constant::ProtoPair(
                    Type::Pair(Type::Integer.into(), Type::Bool.into()),
                    Type::Integer,
                    Constant::ProtoPair(
                        Type::Integer,
                        Type::Bool,
                        Constant::Integer(11.into()).into(),
                        Constant::Bool(true).into(),
                    )
                    .into(),
                    Constant::Integer(11.into()).into(),
                )
                .into(),
                uniq_id: next_uniq_id(),
            },
        };

        let actual_program: Program<Name> = Program::unflat(&bytes).unwrap();

        assert_eq!(actual_program, expected_program)
    }

    #[test]
    fn flat_decode_integer() {
        let bytes = vec![
            0b00001011, 0b00010110, 0b00100001, 0b01001000, 0b00000101, 0b10000001,
        ];

        let expected_program = Program {
            version: (11, 22, 33),
            term: Term::Constant {
                value: Constant::Integer(11.into()).into(),
                uniq_id: next_uniq_id(),
            },
        };

        let actual_program: Program<Name> = Program::unflat(&bytes).unwrap();

        assert_eq!(actual_program, expected_program)
    }

    #[test]
    fn unflat_string_escape() {
        let cbor = "490000004901015c0001";

        let program =
            Program::<DeBruijn>::from_hex(cbor, &mut Vec::new(), &mut Vec::new()).unwrap();

        assert_eq!(
            program.to_pretty().as_str(),
            indoc! { r#"
              (program
                0.0.0
                (con string "\\")
              )"#}
        );
    }

    #[test]
    fn uplc_parser_string_escape() {
        let source = indoc! { r#"
            (program
              0.0.0
              (con string "\n\t\\\"\'\r")
            )"#};

        let program = parser::program(source).unwrap();

        assert_eq!(program.to_pretty(), source);
    }

    /// A malformed program must still report WHERE decoding failed, not just
    /// what failed. `Program::decode` runs the fast iterative decoder and, on
    /// error, rewinds and re-runs the recursive one purely to build that
    /// message. Flat is bit-addressed, so the rewind has to restore the bit
    /// offset as well as the byte offset — restoring `pos` alone resumes on
    /// the right byte at the wrong bit and the path it reports is nonsense.
    #[test]
    fn flat_decode_error_reports_the_path_into_the_term() {
        let source = indoc! { r#"
            (program
              1.0.0
              (lam x (delay (con integer 11)))
            )"#};

        let program = parser::program(source).unwrap();
        let bytes = program.to_flat().unwrap();

        // Cut the term short so decoding runs off the end inside the `delay`.
        let truncated = &bytes[..bytes.len() - 6];

        let error = Program::<DeBruijn>::unflat(truncated)
            .expect_err("a truncated program must not decode");

        // The state log names the constructors it walked into before the
        // failure. Reaching the `lam` proves the rewind landed on the exact
        // bit the term starts at.
        let message = format!("{error}");

        assert!(
            message.contains("(lam") && message.contains("(delay"),
            "decode error lost the path into the term: {message}"
        );
    }
}
