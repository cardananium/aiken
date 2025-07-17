use std::sync::atomic::{AtomicIsize, Ordering};

pub static GLOBAL_UNIQ_GENERATOR: AtomicIsize = AtomicIsize::new(0);

pub fn next_uniq_id() -> isize {
    GLOBAL_UNIQ_GENERATOR.fetch_add(1, Ordering::Relaxed) + 1
}

// pub(crate) fn reset_uniq_generator() {
//     GLOBAL_UNIQ_GENERATOR.store(0, Ordering::Relaxed);
// }