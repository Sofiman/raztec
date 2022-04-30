use raztec::{self, writer::AztecWriter};
use std::io::Write;

fn main() {
    let mut writer = AztecWriter::new(1);
    let code: Vec<u8> = 
        "10011111 00100000 01011100 00011100 00101010 10101000 10010100 11110000"
        .split(' ').flat_map(|val| u8::from_str_radix(val, 2)).collect();
    writer.write_all(&code).unwrap();

    //writer.write_all(&[0b10011001; 32][..]).unwrap();
    println!("{}", writer);
    let mut code = writer.into_aztec();
    code.invert();
    println!("{}", code);
}
