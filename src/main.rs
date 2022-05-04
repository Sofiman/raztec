use std::time::Instant;

use raztec::writer::AztecCodeBuilder;

fn main() {
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)
        .unwrap();
    let start = Instant::now();
    let code = AztecCodeBuilder::new()
        .append("buffer: [")
        .append_bytes(input.trim().as_bytes())
        .append("]").build();
    /*
    let code = AztecCodeBuilder::new().error_correction(23)
        .append("Hello").append(", ").append("World! 0123456789").build();
    */
    /*
    let code = AztecCodeBuilder::new()
        .append("7551357a-5d63-4c6e-81ae-62dca7967f06").build();
    */
    /*
    let code = AztecCodeBuilder::new()
        .append("Test Lower Case Toggle For Shift").build();
    */
    println!("Successfully generated Aztec Code in {:?}", start.elapsed());
    println!("{}", code);
}
