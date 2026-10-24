pub mod ast;
pub mod builder;
pub mod builtins;
mod debruijn;
mod debug;
pub mod flat;
pub mod machine;
pub mod manual_machine;
pub mod optimize;
pub mod parser;
mod pretty;
pub mod tx;

pub mod global_uniq;

use pallas_codec::minicbor;
pub use pallas_crypto::hash::Hash;
pub use pallas_primitives::{
    BigInt, BoundedBytes, Constr, Error, Fragment, KeyValuePairs, MaybeIndefArray, PlutusData,
    conway::{Language, PostAlonzoTransactionOutput, TransactionInput, TransactionOutput, Value},
};
pub use tx::redeemer_tag_to_string;

pub fn plutus_data(bytes: &[u8]) -> Result<PlutusData, Error> {
    PlutusData::decode_fragment(bytes)
}

// ---------------- re-encode Plutus Data

// The 'serialise_data' builtin must re-encode PlutusData in a 'canonical' way which mainly differs
// in few ways compared to the current Pallas' encoding:
//
// 1. Arrays are always encoded using indefinite arrays, except when empty. When empty, they're
//    always encoded using definite length.
// 2. Maps are always encoded with definite length, even when empty.
// 3. Constr fields follow the same rules as arrays.
//
// Data nests as deep as its CBOR says (see the `Drop` impl on `Term` in `ast.rs` for why depth
// matters), so the encoder keeps what is left to write on an explicit stack: pre-order, with the
// break that closes an indefinite array waiting behind its items.
pub fn plutus_data_to_bytes(data: &PlutusData) -> Vec<u8> {
    enum Step<'a> {
        Data(&'a PlutusData),
        End,
    }

    fn array<'a>(
        array: &'a MaybeIndefArray<PlutusData>,
        e: &mut minicbor::Encoder<&mut Vec<u8>>,
        steps: &mut Vec<Step<'a>>,
    ) {
        match array {
            MaybeIndefArray::Def(vec) | MaybeIndefArray::Indef(vec) => {
                // Mimics default haskell list encoding from cborg:
                // We use indef array for non-empty arrays but definite 0-length array for empty
                if vec.is_empty() {
                    e.array(0).unwrap();
                } else {
                    e.begin_array().unwrap();
                    steps.push(Step::End);
                    steps.extend(vec.iter().rev().map(Step::Data));
                }
            }
        }
    }

    let mut buffer = Vec::new();
    let mut e = minicbor::Encoder::new(&mut buffer);
    let mut steps = vec![Step::Data(data)];

    while let Some(step) = steps.pop() {
        let data = match step {
            Step::End => {
                e.end().unwrap();
                continue;
            }
            Step::Data(data) => data,
        };

        match data {
            PlutusData::Constr(constr) => {
                e.tag(minicbor::data::Tag::new(constr.tag)).unwrap();
                if constr.tag == 102 {
                    e.array(2).unwrap();
                    e.encode(constr.any_constructor.unwrap_or_default())
                        .unwrap();
                }
                array(&constr.fields, &mut e, &mut steps);
            }
            PlutusData::Map(kvs) => match kvs {
                KeyValuePairs::Def(pairs) | KeyValuePairs::Indef(pairs) => {
                    // Force definite encoding for maps.
                    e.map(pairs.len() as u64).unwrap();
                    for (k, v) in pairs.iter().rev() {
                        steps.push(Step::Data(v));
                        steps.push(Step::Data(k));
                    }
                }
            },
            PlutusData::Array(items) => array(items, &mut e, &mut steps),
            PlutusData::BoundedBytes(bytes) => {
                e.encode(bytes)
                    .expect("failed to encode BoundedBytes in a bytes buffer?!");
            }
            PlutusData::BigInt(i) => {
                e.encode(i)
                    .expect("failed to encode BigInt in a bytes buffer?!");
            }
        }
    }

    buffer
}
