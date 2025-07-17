use crate::{
    ast::{Constant, Name, Term, Type},
    builtins::DefaultFunction, global_uniq::next_uniq_id,
};
use pallas_primitives::alonzo::PlutusData;
use std::rc::Rc;

pub const CONSTR_FIELDS_EXPOSER: &str = "__constr_fields_exposer";
pub const CONSTR_INDEX_EXPOSER: &str = "__constr_index_exposer";
pub const EXPECT_ON_LIST: &str = "__expect_on_list";
pub const INNER_EXPECT_ON_LIST: &str = "__inner_expect_on_list";
pub const INDICES_CONVERTER: &str = "__indices_converter";

impl<T> Term<T>
where
    T: std::fmt::Debug,
{
    // Terms
    pub fn apply(self, arg: Self) -> Self {
        Term::Apply {
            function: self.into(),
            argument: arg.into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn force(self) -> Self {
        Term::Force {
            body: self.into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn delay(self) -> Self {
        Term::Delay {
            body: self.into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn constr(tag: usize, fields: Vec<Term<T>>) -> Self {
        Term::Constr {
            tag,
            fields,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn case(self, branches: Vec<Term<T>>) -> Self {
        Term::Case {
            constr: self.into(),
            branches,
            uniq_id: next_uniq_id(),
        }
    }

    // Primitives
    pub fn integer(i: num_bigint::BigInt) -> Self {
        Term::Constant {
            value: Constant::Integer(i).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn string(s: impl ToString) -> Self {
        Term::Constant {
            value: Constant::String(s.to_string()).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn byte_string(b: Vec<u8>) -> Self {
        Term::Constant {
            value: Constant::ByteString(b).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn bls12_381_g1(b: blst::blst_p1) -> Self {
        Term::Constant {
            value: Constant::Bls12_381G1Element(b.into()).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn bls12_381_g2(b: blst::blst_p2) -> Self {
        Term::Constant {
            value: Constant::Bls12_381G2Element(b.into()).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn bool(b: bool) -> Self {
        Term::Constant {
            value: Constant::Bool(b).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn unit() -> Self {
        Term::Constant {
            value: Constant::Unit.into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn data(d: PlutusData) -> Self {
        Term::Constant {
            value: Constant::Data(d).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn empty_list() -> Self {
        Term::Constant {
            value: Constant::ProtoList(Type::Data, vec![]).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn list_values(vals: Vec<Constant>) -> Self {
        Term::Constant {
            value: Constant::ProtoList(Type::Data, vals).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn int_values(vals: Vec<Constant>) -> Self {
        Term::Constant {
            value: Constant::ProtoList(Type::Integer, vals).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn empty_map() -> Self {
        Term::Constant {
            value: Constant::ProtoList(Type::Pair(Type::Data.into(), Type::Data.into()), vec![]).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn map_values(vals: Vec<Constant>) -> Self {
        Term::Constant {
            value: Constant::ProtoList(Type::Pair(Type::Data.into(), Type::Data.into()), vals).into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn pair_values(fst_val: Constant, snd_val: Constant) -> Self {
        Term::Constant {
            value: Constant::ProtoPair(Type::Data, Type::Data, fst_val.into(), snd_val.into()).into(),
            uniq_id: next_uniq_id(),
        }
    }

    // This section contains builders for builtins from default functions
    // Theses are in _alphabetical order_
    // The naming convention almost follows PascalCase -> snake_case
    // Exceptions include the use of `un`.

    pub fn add_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::AddInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn append_bytearray() -> Self {
        Term::Builtin {
            fun:DefaultFunction::AppendByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn append_string() -> Self {
        Term::Builtin {
            fun:DefaultFunction::AppendString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn b_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::BData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn blake2b_224() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Blake2b_224,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn blake2b_256() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Blake2b_256,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn bls12_381_g1_add() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G1_Add,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g1_neg() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G1_Neg,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g1_scalar_mul() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G1_ScalarMul,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g1_equal() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G1_Equal,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g1_compress() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G1_Compress,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g1_uncompress() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G1_Uncompress,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g1_hash_to_group() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G1_HashToGroup,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g2_add() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G2_Add,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g2_neg() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G2_Neg,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g2_scalar_mul() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G2_ScalarMul,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g2_equal() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G2_Equal,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g2_compress() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G2_Compress,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g2_uncompress() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G2_Uncompress,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_g2_hash_to_group() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_G2_HashToGroup,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_miller_loop() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_MillerLoop,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_mul_ml_result() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_MulMlResult,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn bls12_381_final_verify() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Bls12_381_FinalVerify,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn choose_data(
        self,
        constr_case: Self,
        map_case: Self,
        array_case: Self,
        int_case: Self,
        bytes_case: Self,
    ) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseData,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(constr_case)
            .apply(map_case)
            .apply(array_case)
            .apply(int_case)
            .apply(bytes_case)
    }

    pub fn choose_list(self, then_term: Self, else_term: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseList,
            uniq_id: next_uniq_id(),
        }
            .force()
            .force()
            .apply(self)
            .apply(then_term)
            .apply(else_term)
    }

    pub fn choose_unit(self, then_term: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseUnit,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(then_term)
    }

    pub fn cons_bytearray() -> Self {
        Term::Builtin {
            fun:DefaultFunction::ConsByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn constr_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::ConstrData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn decode_utf8() -> Self {
        Term::Builtin {
            fun:DefaultFunction::DecodeUtf8,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn div_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::DivideInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn divide_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::DivideInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn encode_utf8() -> Self {
        Term::Builtin {
            fun:DefaultFunction::EncodeUtf8,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn equals_bytestring() -> Self {
        Term::Builtin {
            fun:DefaultFunction::EqualsByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn equals_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::EqualsData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn equals_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::EqualsInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn equals_string() -> Self {
        Term::Builtin {
            fun:DefaultFunction::EqualsString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn fst_pair() -> Self {
        Term::Builtin {
            fun:DefaultFunction::FstPair,
            uniq_id: next_uniq_id(),
        }
            .force()
            .force()
    }

    pub fn head_list() -> Self {
        Term::Builtin {
            fun:DefaultFunction::HeadList,
            uniq_id: next_uniq_id(),
        }
            .force()
    }

    pub fn i_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::IData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn if_then_else(self, then_term: Self, else_term: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::IfThenElse,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(then_term)
            .apply(else_term)
    }

    pub fn index_bytearray() -> Self {
        Term::Builtin {
            fun:DefaultFunction::IndexByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn keccak_256() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Keccak_256,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn length_of_bytearray() -> Self {
        Term::Builtin {
            fun:DefaultFunction::LengthOfByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn less_than_bytearray() -> Self {
        Term::Builtin {
            fun:DefaultFunction::LessThanByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn less_than_equals_bytearray() -> Self {
        Term::Builtin {
            fun:DefaultFunction::LessThanEqualsByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn less_than_equals_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::LessThanEqualsInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn less_than_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::LessThanInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn list_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::ListData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn map_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::MapData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn mk_cons() -> Self {
        Term::Builtin {
            fun:DefaultFunction::MkCons,
            uniq_id: next_uniq_id(),
        }
            .force()
    }

    pub fn mk_pair_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::MkPairData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn mod_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::ModInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn multiply_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::MultiplyInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn quotient_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::QuotientInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn remainder_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::RemainderInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn sha2_256() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Sha2_256,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn sha3_256() -> Self {
        Term::Builtin {
            fun:DefaultFunction::Sha3_256,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn slice_bytearray() -> Self {
        Term::Builtin {
            fun:DefaultFunction::SliceByteString,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn snd_pair() -> Self {
        Term::Builtin {
            fun:DefaultFunction::SndPair,
            uniq_id: next_uniq_id(),
        }
            .force()
            .force()
    }

    pub fn subtract_integer() -> Self {
        Term::Builtin {
            fun:DefaultFunction::SubtractInteger,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn tail_list() -> Self {
        Term::Builtin {
            fun:DefaultFunction::TailList,
            uniq_id: next_uniq_id(),
        }
            .force()
    }

    pub fn un_b_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::UnBData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn un_i_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::UnIData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn unconstr_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::UnConstrData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn unlist_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::UnListData,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn unmap_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::UnMapData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn verify_ecdsa_secp256k1_signature() -> Self {
        Term::Builtin {
            fun:DefaultFunction::VerifyEcdsaSecp256k1Signature,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn verify_ed25519_signature() -> Self {
        Term::Builtin {
            fun:DefaultFunction::VerifyEd25519Signature,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn verify_schnorr_secp256k1_signature() -> Self {
        Term::Builtin {
            fun:DefaultFunction::VerifySchnorrSecp256k1Signature,
            uniq_id: next_uniq_id(),
        }
    }

    // Unused bultins
    pub fn mk_nil_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::MkNilData,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn mk_nil_pair_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::MkNilPairData,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn null_list() -> Self {
        Term::Builtin {
            fun:DefaultFunction::NullList,
            uniq_id: next_uniq_id(),
        }
    }
    pub fn serialise_data() -> Self {
        Term::Builtin {
            fun:DefaultFunction::SerialiseData,
            uniq_id: next_uniq_id(),
        }
    }

    pub fn write_bits() -> Self {
        Term::Builtin {
            fun:DefaultFunction::WriteBits,
            uniq_id: next_uniq_id(),
        }
    }
}

impl<T> Term<T>
where
    T: std::fmt::Debug,
{
    pub fn delayed_choose_data(
        self,
        constr_case: Self,
        map_case: Self,
        array_case: Self,
        int_case: Self,
        bytes_case: Self,
    ) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseData,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(constr_case.delay())
            .apply(map_case.delay())
            .apply(array_case.delay())
            .apply(int_case.delay())
            .apply(bytes_case.delay())
            .force()
    }

    pub fn delayed_choose_list(self, then_term: Self, else_term: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseList,
            uniq_id: next_uniq_id(),
        }
            .force()
            .force()
            .apply(self)
            .apply(then_term.delay())
            .apply(else_term.delay())
            .force()
    }

    /// Note the otherwise is expected to be a delayed term cast to a Var
    pub fn delay_empty_choose_list(self, empty: Self, otherwise: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseList,
            uniq_id: next_uniq_id(),
        }
            .force()
            .force()
            .apply(self)
            .apply(empty.delay())
            .apply(otherwise)
            .force()
    }

    /// Note the otherwise is expected to be a delayed term cast to a Var
    pub fn delay_filled_choose_list(self, otherwise: Self, filled: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseList,
            uniq_id: next_uniq_id(),
        }
            .force()
            .force()
            .apply(self)
            .apply(otherwise)
            .apply(filled.delay())
            .force()
    }

    pub fn delayed_choose_unit(self, then_term: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::ChooseUnit,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(then_term.delay())
            .force()
    }

    pub fn delayed_if_then_else(self, then_term: Self, else_term: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::IfThenElse,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(then_term.delay())
            .apply(else_term.delay())
            .force()
    }

    /// Note the otherwise is expected to be a delayed term cast to a Var
    pub fn delay_true_if_then_else(self, then: Self, otherwise: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::IfThenElse,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(then.delay())
            .apply(otherwise)
            .force()
    }

    /// Note the otherwise is expected to be a delayed term cast to a Var
    pub fn delay_false_if_then_else(self, otherwise: Self, alternative: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::IfThenElse,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(self)
            .apply(otherwise)
            .apply(alternative.delay())
            .force()
    }

    pub fn delayed_trace(self, msg_term: Self) -> Self {
        Term::Builtin {
            fun:DefaultFunction::Trace,
            uniq_id: next_uniq_id(),
        }
            .force()
            .apply(msg_term)
            .apply(self.delay())
            .force()
    }

    // Misc.
    pub fn repeat_tail_list(self, repeat: usize) -> Self {
        let mut term = self;

        for _ in 0..repeat {
            term = Term::tail_list().apply(term);
        }

        term
    }
}

impl Term<Name> {
    pub fn lambda(self, parameter_name: impl ToString) -> Self {
        Term::Lambda {
            parameter_name: Name::text(parameter_name).into(),
            body: self.into(),
            uniq_id: next_uniq_id(),
        }
    }

    pub fn var(name: impl ToString) -> Self {
        Term::Var {
            name: Name::text(name).into(),
            uniq_id: next_uniq_id(),
        }
    }

    // Misc.
    pub fn constr_fields_exposer(self) -> Self {
        self.lambda(CONSTR_FIELDS_EXPOSER).apply(
            Term::snd_pair()
                .apply(Term::unconstr_data().apply(Term::var("__constr_var")))
                .lambda("__constr_var"),
        )
    }

    pub fn constr_index_exposer(self) -> Self {
        self.lambda(CONSTR_INDEX_EXPOSER).apply(
            Term::fst_pair()
                .apply(Term::unconstr_data().apply(Term::var("__constr_var")))
                .lambda("__constr_var"),
        )
    }

    pub fn data_list_to_integer_list(self) -> Self {
        self.lambda(INDICES_CONVERTER)
            .apply(Term::var(INDICES_CONVERTER).apply(Term::var(INDICES_CONVERTER)))
            .lambda(INDICES_CONVERTER)
            .apply(
                Term::var("xs")
                    .delayed_choose_list(
                        Term::int_values(vec![]),
                        Term::mk_cons()
                            .apply(Term::var("x"))
                            .apply(
                                Term::var(INDICES_CONVERTER)
                                    .apply(Term::var(INDICES_CONVERTER))
                                    .apply(Term::var("rest")),
                            )
                            .lambda("rest")
                            .apply(Term::tail_list().apply(Term::var("xs")))
                            .lambda("x")
                            .apply(
                                Term::un_i_data().apply(Term::head_list().apply(Term::var("xs"))),
                            ),
                    )
                    .lambda("xs")
                    .lambda(INDICES_CONVERTER),
            )
    }

    /// Introduce a let-binding for a given term. The callback receives a Term::Var
    /// whose name matches the given 'var_name'. Handy to re-use a same var across
    /// multiple lambda expressions.
    ///
    /// ## Example
    ///
    ///
    ///
    /// use uplc::ast::Term
    /// let value = Term::var("thing");
    ///
    /// value.as_var("__val", |val| {
    ///   val.do_something()
    ///      .do_another_thing()
    /// })
    ///
    pub fn as_var<F>(self, var_name: &str, callback: F) -> Term<Name>
    where
        F: FnOnce(Rc<Name>) -> Term<Name>,
    {
        callback(Name::text(var_name).into())
            .lambda(var_name)
            .apply(self)
    }

    /// Continue a computation provided that the current term is a Data-wrapped integer.
    /// The 'callback' receives an integer constant Term as argument.
    pub fn choose_data_integer<F>(var: Rc<Name>, callback: F, otherwise: &Term<Name>) -> Self
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        Term::Var {
            name: var.clone(),
            uniq_id: next_uniq_id(),
        }
            .choose_data(
                otherwise.clone(),
                otherwise.clone(),
                otherwise.clone(),
                callback(Term::un_i_data().apply(Term::Var {
                    name: var.clone(),
                    uniq_id: next_uniq_id(),
                })).delay(),
                otherwise.clone(),
            )
            .force()
    }

    /// Continue a computation provided that the current term is a Data-wrapped
    /// bytearray. The 'callback' receives a bytearray constant Term as argument.
    pub fn choose_data_bytearray<F>(var: Rc<Name>, callback: F, otherwise: &Term<Name>) -> Self
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        Term::Var {
            name: var.clone(),
            uniq_id: next_uniq_id(),
        }
            .choose_data(
                otherwise.clone(),
                otherwise.clone(),
                otherwise.clone(),
                otherwise.clone(),
                callback(Term::un_b_data().apply(Term::Var {
                    name: var.clone(),
                    uniq_id: next_uniq_id(),
                })).delay(),
            )
            .force()
    }

    /// Continue a computation provided that the current term is a Data-wrapped
    /// list. The 'callback' receives a ProtoList Term as argument.
    pub fn choose_data_list<F>(var: Rc<Name>, callback: F, otherwise: &Term<Name>) -> Self
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        Term::Var {
            name: var.clone(),
            uniq_id: next_uniq_id(),
        }
            .choose_data(
                otherwise.clone(),
                otherwise.clone(),
                callback(Term::unlist_data().apply(Term::Var {
                    name: var.clone(),
                    uniq_id: next_uniq_id(),
                })).delay(),
                otherwise.clone(),
                otherwise.clone(),
            )
            .force()
    }

    /// Continue a computation provided that the current term is a Data-wrapped
    /// list. The 'callback' receives a ProtoMap Term as argument.
    pub fn choose_data_map<F>(var: Rc<Name>, callback: F, otherwise: &Term<Name>) -> Self
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        Term::Var {
            name: var.clone(),
            uniq_id: next_uniq_id(),
        }
            .choose_data(
                otherwise.clone(),
                callback(Term::unmap_data().apply(Term::Var {
                    name: var.clone(),
                    uniq_id: next_uniq_id(),
                })).delay(),
                otherwise.clone(),
                otherwise.clone(),
                otherwise.clone(),
            )
            .force()
    }

    /// Continue a computation provided that the current term is a Data-wrapped
    /// constr. The 'callback' receives a Data as argument.
    pub fn choose_data_constr<F>(var: Rc<Name>, callback: F, otherwise: &Term<Name>) -> Self
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        Term::Var {
            name: var.clone(),
            uniq_id: next_uniq_id(),
        }
            .choose_data(
                callback(Term::Var {
                    name: var.clone(),
                    uniq_id: next_uniq_id(),
                }).delay(),
                otherwise.clone(),
                otherwise.clone(),
                otherwise.clone(),
                otherwise.clone(),
            )
            .force()
    }

    /// Convert an arbitrary 'term' into a bool term and pass it into a 'callback'.
    /// Continue the execution 'otherwise' with a different branch.
    ///
    /// Note that the 'otherwise' term is expected
    /// to be a delayed term.
    pub fn unwrap_bool_or<F>(self, callback: F, otherwise: &Term<Name>) -> Term<Name>
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        Term::unconstr_data()
            .apply(self)
            .as_var("__pair__", |pair| {
                Term::snd_pair()
                    .apply(Term::Var {
                        name: pair.clone(),
                        uniq_id: next_uniq_id(),
                    })
                    .delay_empty_choose_list(
                        Term::less_than_equals_integer()
                            .apply(Term::integer(2.into()))
                            .apply(Term::fst_pair().apply(Term::Var {
                                name: pair.clone(),
                                uniq_id: next_uniq_id(),
                            }))
                            .delay_false_if_then_else(
                                otherwise.clone(),
                                callback(
                                    Term::equals_integer()
                                        .apply(Term::integer(1.into()))
                                        .apply(Term::fst_pair().apply(Term::Var {
                                            name: pair.clone(),
                                            uniq_id: next_uniq_id(),
                                        })),
                                ),
                            ),
                        otherwise.clone(),
                    )
            })
    }

    /// Convert an arbitrary 'term' into a unit term and pass it into a 'callback'.
    /// Continue the execution 'otherwise' with a different branch.
    ///
    /// Note that the 'otherwise' term is expected
    /// to be a delayed term.
    pub fn unwrap_void_or<F>(self, callback: F, otherwise: &Term<Name>) -> Term<Name>
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        assert!(matches!(self, Term::Var { .. }));
        Term::equals_integer()
            .apply(Term::integer(0.into()))
            .apply(Term::fst_pair().apply(Term::unconstr_data().apply(self.clone())))
            .delay_true_if_then_else(
                Term::snd_pair()
                    .apply(Term::unconstr_data().apply(self))
                    .delay_empty_choose_list(callback(Term::unit()), otherwise.clone()),
                otherwise.clone(),
            )
    }

    /// Convert an arbitrary 'term' into a pair and pass it into a 'callback'.
    /// Continue the execution 'otherwise' with a different branch.
    ///
    /// Note that the 'otherwise' term is expected
    /// to be a delayed term.
    pub fn unwrap_pair_or<F>(self, callback: F, otherwise: &Term<Name>) -> Term<Name>
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        self.as_var("__list_data", |list| {
            let left = Term::head_list().apply(Term::Var {
                name: list.clone(),
                uniq_id: next_uniq_id(),
            });

            Term::unwrap_tail_or(
                list,
                |tail| {
                    tail.as_var("__tail", |tail| {
                        let right = Term::head_list().apply(Term::Var {
                            name: tail.clone(),
                            uniq_id: next_uniq_id(),
                        });

                        Term::unwrap_tail_or(
                            tail,
                            |leftovers| {
                                leftovers.delay_empty_choose_list(
                                    callback(Term::mk_pair_data().apply(left).apply(right)),
                                    otherwise.clone(),
                                )
                            },
                            otherwise,
                        )
                    })
                },
                otherwise,
            )
        })
    }

    /// Continue with the tail of a list, if any; or fallback 'otherwise'.
    ///
    /// Note that the 'otherwise' term is expected
    /// to be a delayed term.
    pub fn unwrap_tail_or<F>(var: Rc<Name>, callback: F, otherwise: &Term<Name>) -> Term<Name>
    where
        F: FnOnce(Term<Name>) -> Term<Name>,
    {
        Term::Var {
            name: var.clone(),
            uniq_id: next_uniq_id(),
        }
            .delay_filled_choose_list(
                otherwise.clone(),
                callback(Term::tail_list().apply(Term::Var {
                    name: var.clone(),
                    uniq_id: next_uniq_id(),
                })),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Data, Name, NamedDeBruijn, Program, Term},
        builder::Constant,
        machine::{Error, cost_model::ExBudget},
        optimize::interner::CodeGenInterner,
    };

    fn quick_eval(term: Term<Name>) -> Result<Term<NamedDeBruijn>, Error> {
        let version = (1, 0, 0);
        let mut program = Program { version, term };
        CodeGenInterner::new().program(&mut program);
        program
            .to_named_debruijn()
            .expect("failed to convert program to NamedDeBruijn")
            .eval(ExBudget::default())
            .result()
    }

    #[test]
    fn unwrap_bool_or_false() {
        let result = quick_eval(
            Term::data(Data::constr(0, vec![])).unwrap_bool_or(|b| b, &Term::Error { uniq_id: 0 }.delay()),
        );

        assert_eq!(result, Ok(Term::bool(false)));
    }

    #[test]
    fn unwrap_bool_or_true() {
        let result = quick_eval(
            Term::data(Data::constr(1, vec![])).unwrap_bool_or(|b| b, &Term::Error { uniq_id: 0 }.delay()),
        );

        assert_eq!(result, Ok(Term::bool(true)));
    }

    #[test]
    fn unwrap_bool_or_extra_args() {
        let result = quick_eval(
            Term::data(Data::constr(1, vec![Data::integer(42.into())]))
                .unwrap_bool_or(|b| b, &Term::Error { uniq_id: 0 }.delay()),
        );

        assert_eq!(result, Err(Error::EvaluationFailure));
    }

    #[test]
    fn unwrap_bool_or_invalid_constr_hi() {
        let result = quick_eval(
            Term::data(Data::constr(2, vec![])).unwrap_bool_or(|b| b, &Term::Error { uniq_id: 0 }.delay()),
        );

        assert_eq!(result, Err(Error::EvaluationFailure));
    }

    #[test]
    fn unwrap_tail_or_0_elems() {
        let result = quick_eval(Term::list_values(vec![]).as_var("__tail", |tail| {
            Term::unwrap_tail_or(tail, |p| p, &Term::Error { uniq_id: 0 }.delay())
        }));

        assert_eq!(result, Err(Error::EvaluationFailure));
    }

    #[test]
    fn unwrap_tail_or_1_elem() {
        let result = quick_eval(
            Term::list_values(vec![Constant::Data(Data::integer(1.into()))])
                .as_var("__tail", |tail| {
                    Term::unwrap_tail_or(tail, |p| p, &Term::Error { uniq_id: 0 }.delay())
                }),
        );

        assert_eq!(result, Ok(Term::list_values(vec![])),);
    }

    #[test]
    fn unwrap_tail_or_2_elems() {
        let result = quick_eval(
            Term::list_values(vec![
                Constant::Data(Data::integer(1.into())),
                Constant::Data(Data::integer(2.into())),
            ])
            .as_var("__tail", |tail| {
                Term::unwrap_tail_or(tail, |p| p, &Term::Error { uniq_id: 0 }.delay())
            }),
        );

        assert_eq!(
            result,
            Ok(Term::list_values(vec![Constant::Data(Data::integer(
                2.into()
            ))]))
        );
    }

    #[test]
    fn unwrap_pair_or() {
        let result = quick_eval(
            Term::list_values(vec![
                Constant::Data(Data::integer(14.into())),
                Constant::Data(Data::bytestring(vec![1, 2, 3])),
            ])
            .unwrap_pair_or(|p| p, &Term::Error { uniq_id: 0 }.delay()),
        );

        assert_eq!(
            result,
            Ok(Term::pair_values(
                Constant::Data(Data::integer(14.into())),
                Constant::Data(Data::bytestring(vec![1, 2, 3])),
            ))
        );
    }

    #[test]
    fn unwrap_pair_or_not_enough_args_1() {
        let result = quick_eval(
            Term::list_values(vec![Constant::Data(Data::integer(1.into()))])
                .unwrap_pair_or(|p| p, &Term::Error { uniq_id: 0 }.delay()),
        );

        assert_eq!(result, Err(Error::EvaluationFailure));
    }

    #[test]
    fn unwrap_pair_or_not_enough_args_0() {
        let result =
            quick_eval(Term::list_values(vec![]).unwrap_pair_or(|p| p, &Term::Error { uniq_id: 0 }.delay()));

        assert_eq!(result, Err(Error::EvaluationFailure));
    }

    #[test]
    fn unwrap_pair_or_too_many_args() {
        let result = quick_eval(
            Term::list_values(vec![
                Constant::Data(Data::integer(1.into())),
                Constant::Data(Data::integer(2.into())),
                Constant::Data(Data::integer(3.into())),
            ])
            .unwrap_pair_or(|p| p, &Term::Error { uniq_id: 0 }.delay()),
        );

        assert_eq!(result, Err(Error::EvaluationFailure));
    }

    #[test]
    fn unwrap_void_or_happy() {
        let result = quick_eval(
            Term::data(Data::constr(0, vec![])).as_var("__unit", |unit| {
                Term::Var {
                    name: unit.clone(),
                    uniq_id: 0,
                }
                    .unwrap_void_or(|u| u, &Term::Error { uniq_id: 0 }.delay())
            }),
        );

        assert_eq!(result, Ok(Term::unit()));
    }

    #[test]
    fn unwrap_void_or_wrong_constr() {
        let result = quick_eval(
            Term::data(Data::constr(14, vec![])).as_var("__unit", |unit| {
                Term::Var {
                    name: unit.clone(),
                    uniq_id: 0,
                }
                    .unwrap_void_or(|u| u, &Term::Error { uniq_id: 0 }.delay())
            }),
        );

        assert_eq!(result, Err(Error::EvaluationFailure));
    }

    #[test]
    fn unwrap_void_or_too_many_args() {
        let result = quick_eval(
            Term::data(Data::constr(0, vec![Data::integer(0.into())])).as_var("__unit", |unit| {
                Term::Var {
                    name: unit.clone(),
                    uniq_id: 0,
                }
                    .unwrap_void_or(|u| u, &Term::Error { uniq_id: 0 }.delay())
            }),
        );

        assert_eq!(result, Err(Error::EvaluationFailure));
    }
}
