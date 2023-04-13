pub struct ThreadPool {
    n_threads: usize,
}

impl ThreadPool {
    pub fn new(n_threads: usize) -> Self {
        Self { n_threads }
    }
    pub fn max_count(&self) -> usize {
        self.n_threads
    }

    pub fn execute<F: Fn() -> ()>(&self, f: F) {
        f();
    }

    pub fn join(&self) {}
}

pub struct Once {}

impl Once {
    pub const fn new() -> Self {
        Self {}
    }

    pub fn call_once<F: Fn() -> ()>(&self, f: F) -> Self {
        Self {}
    }
}
