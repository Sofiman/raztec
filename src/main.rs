use raztec::writer::AztecCodeBuilder;

fn main() {
    /*let code = AztecCodeBuilder::new(23)
        .append("Hello").append(", ").append("World!").build();*/
    let code = AztecCodeBuilder::new(23)
        .append("Nino Nakano").build();
    /*
    let code = AztecCodeBuilder::new(23)
        .append("7551357a-5d63-4c6e-81ae-62dca7967f06").build();
    */
    println!("{}", code);
}
