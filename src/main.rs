use std::time::Instant;

use raztec::writer::AztecCodeBuilder;

fn main() {
    let start = Instant::now();
    let code = AztecCodeBuilder::new(23)
        .append("Hello").append(", ").append("World! 0123456789").build();
    /*
    let code = AztecCodeBuilder::new(23)
        .append("7551357a-5d63-4c6e-81ae-62dca7967f06").build();
    */
    println!("Successfully generated Aztec Code in {:?}", start.elapsed());
    println!("{}", code);
}
