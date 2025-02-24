use std::time::Instant;

#[allow(dead_code)]
pub struct Timer {
    last: Instant,
}

#[allow(dead_code)]
impl Timer {
    pub fn new() -> Self {
        Self {
            last: Instant::now(),
        }
    }

    // The `log` method takes a message (anything that implements std::fmt::Debug)
    // and prints the elapsed time since last call, then updates the last timestamp.
    pub fn log<T: std::fmt::Debug>(&mut self, msg: T) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last);
        self.last = now;
        println!("[{:?}] {:?}", elapsed, msg);
    }
}
