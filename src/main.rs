use raztec::{self, writer::AztecCodeBuilder};

fn main() {
    /*let code = AztecCodeBuilder::new(23)
        .append("Hello").append(", ").append("World!").build();*/
    let code = AztecCodeBuilder::new(23)
        .append("Nino Nakano").build();
    println!("{}", code);
}
