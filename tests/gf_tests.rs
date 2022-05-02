use raztec::reed_solomon::{gf::GF, poly::Polynomial};

#[test]
fn test_gf_to_poly() {
    let gf = GF::new(8, 0b101110001); // x^8 + x^4 + x^3 + x^2 + 1
    assert_eq!(gf.get_poly(51), Polynomial::new(&[1, 1, 0, 0, 1, 1]));
}

#[test]
fn test_gf_add() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(7); // 0b111 -> a = x^2 +x + 1
    let b = gf_16.num(3); // 0b011 -> b = x + 1
    assert_eq!((a + b).as_poly(), Polynomial::new(&[0, 0, 1]));
}

#[test]
fn test_gf_add2() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(8); // 0b1000 -> a = x^3
    let b = gf_16.num(5); // 0b0101 -> b = x^2 + 1
    assert_eq!((a + b).as_poly(), Polynomial::new(&[1, 0, 1, 1]));
}

#[test]
fn test_gf_subtract() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(7); // 0b111 -> a = x^2 +x + 1
    let b = gf_16.num(3); // 0b011 -> b = x + 1
    assert_eq!((a - b).as_poly(), Polynomial::new(&[0, 0, 1]));
}

#[test]
fn test_gf_multiply() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(7); // 0b111 -> a = x^2 +x + 1
    let b = gf_16.num(3); // 0b011 -> b = x + 1
    assert_eq!((a * b).as_poly(), Polynomial::new(&[1, 0, 0, 1]));
}

#[test]
fn test_gf_multiply2() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(8); // 0b111 -> a = x^2 +x + 1
    let b = gf_16.num(5); // 0b011 -> b = x + 1
    assert_eq!((a * b).as_poly(), Polynomial::new(&[0, 1, 1, 1]));
}

#[test]
fn test_gf_divide() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(7); // 0b111 -> a = x^2 +x + 1
    let b = gf_16.num(3); // 0b011 -> b = x + 1
    assert_eq!((a / b).as_poly(), Polynomial::new(&[0, 0, 1, 1]));
}

#[test]
fn test_gf_divide2() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(8); // 0b111 -> a = x^2 +x + 1
    let b = gf_16.num(5); // 0b011 -> b = x + 1
    assert_eq!((a / b).as_poly(), Polynomial::new(&[1, 1, 1, 0]));
}

#[test]
fn test_gf_inverse() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(3); // 0b011 -> a = x + 1
    assert_eq!(a.inv().as_poly(), Polynomial::new(&[0, 1, 1, 1]));
}

#[test]
fn test_gf_inverse2() {
    let gf_16 = GF::new(4, 0b10011);
    let a = gf_16.num(5); // 0b011 -> a = x + 1
    assert_eq!(a.inv().as_poly(), Polynomial::new(&[1, 1, 0, 1]));
}
