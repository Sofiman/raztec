use raztec::{self, writer::AztecCodeBuilder};

fn main() {
    let mut writer = AztecCodeBuilder::new(23);
    writer.append("Robomatics");
    let mut code = writer.build();
    code.invert();
    println!("{}", code);
}
