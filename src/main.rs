use raztec::{self, writer::AztecWriter};
use std::io::Write;

fn main() {
    let mut writer = AztecWriter::new(15);
    writer.write_all(&[0b10011001; 13][..]).unwrap();
    let mut code = writer.into_aztec();
    code.invert();
    println!("{}", code);
}
