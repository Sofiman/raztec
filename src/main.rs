use std::time::Instant;

use raztec::writer::AztecCodeBuilder;

fn main() {
    let start = Instant::now();
    /*
    let code = AztecCodeBuilder::new()
        .append("LOREM IPSUM DOLOR SIT AMET CONSECTETUR ADIPISCING ELIT NUNC VESTIBULUM NUNC SED LECTUS SCELERISQUE LAOREET DONEC PULVINAR QUAM A ALIQUAM TINCIDUNT SUSPENDISSE LAOREET TINCIDUNT IPSUM PORTA INTERDUM AUGUE CONSECTETUR A MORBI FERMENTUM ULTRICIES SUSCIPIT IN SUSCIPIT METUS SED SOLLICITUDIN RUTRUM VELIT NISL IMPERDIET URNA A ACCUMSAN MI MAURIS VITAE ENIM SED MOLLIS ENIM IPSUM QUIS CONVALLIS IPSUM VIVERRA VITAE AENEAN EU ELIT AC ANTE ELEIFEND VARIUS CRAS UT NISI QUAM NULLAM ORNARE PORTA ODIO EU PORTA VELIT PULVINAR VEL ETIAM LOREM IPSUM DOLOR SIT AMET CONSECTETUR ADIPISCING ELIT NUNC VESTIBULUM NUNC SED LECTUS SCELERISQUE LAOREET DONEC PULVINAR QUAM A ALIQUAM TINCIDUNT SUSPENDISSE LAOREET TINCIDUNT IPSUM PORTA INTERDUM AUGUE CONSECTETUR A MORBI FERMENTUM ULTRICIES").build().unwrap();
    */
    let code = AztecCodeBuilder::new().error_correction(23)
        .append("Hello").append(", ").append("World! 0123456789")
        .build().unwrap();
    /*
    let code = AztecCodeBuilder::new()
        .append("7551357a-5d63-4c6e-81ae-62dca7967f06").build().unwrap();
    */
    /*
    let code = AztecCodeBuilder::new()
        .append("Test Lower Case Toggle For Shift").build().unwrap();
    */
    /*
    let code = AztecCodeBuilder::new()
        .append("!\"#$%&'()*+,-./:;<=>?[]{}@\\^_`|~\n").build().unwrap();
    */
    /*
    let code = AztecCodeBuilder::new()
        //.append_bytes(&[182])
        .append_eci(26) // \000020
        /*
        .append_bytes("ワンピース".as_bytes())*/
        .append_bytes(&[182]).build().unwrap();
    */
    println!("Successfully generated Aztec Code in {:?}", start.elapsed());
    println!("{}", code);
}
