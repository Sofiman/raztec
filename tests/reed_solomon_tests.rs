use raztec::reed_solomon::{ReedSolomonEncoder};

#[test]
fn test_generate_check_codes(){
    let encoder = ReedSolomonEncoder::new(4, 0b10011);
    let check_codes = encoder.generate_check_codes(&[0, 0b1001], 5);
    assert_eq!(check_codes, [12, 2, 3, 1, 9]);
}

#[test]
fn test_generate_check_codes2(){
    let encoder = ReedSolomonEncoder::new(4, 0b10011);
    let check_codes = encoder.generate_check_codes(&[0, 0b0100], 5);
    assert_eq!(check_codes, [0b1010, 0b0011, 0b1011, 0b1000, 0b0100]);
}

#[test]
fn test_generate_check_codes3(){
    let encoder = ReedSolomonEncoder::new(4, 0b10011);
    let check_codes = encoder.generate_check_codes(&[0b0101, 0b1010], 5);
    assert_eq!(check_codes, [0b1110, 0b0111, 0b0101, 0b0000, 0b1011]);
}

#[test]
fn test_generate_check_codes4(){
    let encoder = ReedSolomonEncoder::new(4, 0b10011);
    let check_codes = encoder.generate_check_codes(&[0b1111, 0b0000], 5);
    assert_eq!(check_codes, [0b0111, 0b1000, 0b0111, 0b1001, 0b0011]);
}

#[test]
fn test_generate_check_big(){
    let encoder = ReedSolomonEncoder::new(6, 0b1000011);
    let check_codes = encoder.generate_check_codes(&[39, 50, 1, 28, 7, 2, 42, 40, 37, 15], 7);
    assert_eq!(check_codes, [44, 29, 43, 52, 49, 22, 15]);
}
