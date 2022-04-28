use raztec::{self, writer::AztecWriter};
use std::io::Write;

fn main() {
    let mut writer = AztecWriter::new(3);
    writer.write_all(&[0b10011001; 32][..]).unwrap();
    println!("{}", writer);
    println!();
    let mut code = writer.into_aztec();
    code.invert();
    println!("{}", code);
}
