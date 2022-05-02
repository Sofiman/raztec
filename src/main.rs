use raztec::{self, writer::AztecCodeBuilder};

fn main() {
    let code = AztecCodeBuilder::new(23)
        .append("4556").build();
    println!("{}", code);
}
