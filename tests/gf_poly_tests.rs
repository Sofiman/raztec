use raztec::reed_solomon::{gf::GF, gf_poly::GFPoly};

#[test]
fn test_gf_poly_deg_zero(){
    let gf: GF = GF::new(4, 0b10011);
    assert_eq!(GFPoly::zero(&gf).deg(), std::isize::MIN);
}

#[test]
fn test_gf_poly_deg_normal(){
    let gf: GF = GF::new(4, 0b10011);
    assert_eq!(GFPoly::from_nums(&gf, &[4, 5]).deg(), 1);
}

#[test]
fn test_gf_poly_deg_padding_zeros(){
    let gf: GF = GF::new(4, 0b10011);
    assert_eq!(GFPoly::from_nums(&gf, &[1, 0, 0, 0, 0, 0, 0]).deg(), 0);
}

#[test]
fn test_gf_poly_deg2(){
    let gf: GF = GF::new(4, 0b10011);
    assert_eq!(GFPoly::from_nums(&gf, &[1, 0, 5, 0, 3]).deg(), 4);
}

#[test]
fn test_gf_poly_equality(){
    let gf: GF = GF::new(4, 0b10011);
    assert_eq!(
        GFPoly::from_nums(&gf, &[1, 0, 3, 4]),
        GFPoly::from_nums(&gf, &[1, 0, 3, 4]));
}

#[test]
fn test_gf_poly_equality_padding(){
    let gf: GF = GF::new(4, 0b10011);
    assert_eq!(
        GFPoly::from_nums(&gf, &[2, 2, 0, 1, 0, 0, 0]),
        GFPoly::from_nums(&gf, &[2, 2, 0, 1]));
}

#[test]
fn test_gf_poly_non_equality(){
    let gf: GF = GF::new(4, 0b10011);
    assert_ne!(GFPoly::from_nums(&gf, &[0, 1, 0, 1]), GFPoly::zero(&gf));
}

#[test]
fn test_gf_poly_evaluate(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[2, 8, 3]);
    // 2 * 2 +  8 * 2^2  +  3 * 2^3 = 60
    // (2⊕2) ⊕ (8*(2*2)) ⊕ (3*(2*(2*2))) = 9
    assert_eq!(a.eval(gf.num(2)), gf.num(9));
}

#[test]
fn test_gf_poly_index(){
    let gf: GF = GF::new(4, 0b10011);
    assert_eq!(GFPoly::from_nums(&gf, &[7, 8, 2, 3])[2], gf.num(2));
}

#[test]
fn test_gf_poly_add_zero(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[1, 2] /* + 0 */);
    assert_eq!(a.clone() + GFPoly::zero(&gf), a);
}

#[test]
fn test_gf_poly_add(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf,           &[3, 1, 2, 6]);
    let b = GFPoly::from_nums(&gf,   /* + */ &[0, 1, 4]);
    assert_eq!(a + b, GFPoly::from_nums(&gf, &[3, 0, 6, 6]));
}

#[test]
fn test_gf_poly_add2(){
    let gf: GF = GF::new(6, 0b1000011);
    let a = GFPoly::from_nums(&gf,           &[6, 5, 1, 1, 1, 0]);
    let b = GFPoly::from_nums(&gf,   /* + */ &[2, 0, 2, 6, 4, 0, 0, 0, 6]);
    assert_eq!(a + b, GFPoly::from_nums(&gf, &[4, 5, 3, 7, 5, 0, 0, 0, 6]));
}

#[test]
fn test_gf_poly_add3(){
    let gf: GF = GF::new(6, 0b1000011);
    let a = GFPoly::from_nums(&gf,           &[1, 1, 1, 0, 0, 1]);
    let b = GFPoly::from_nums(&gf,   /* + */ &[5, 0, 2]);
    assert_eq!(a + b, GFPoly::from_nums(&gf, &[4, 1, 3, 0, 0, 1]));
}

#[test]
fn test_gf_poly_subtract(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf,           &[5, 0, 1, 2]);
    let b = GFPoly::from_nums(&gf,   /* - */ &[0, 5, 3]);
    assert_eq!(a - b, GFPoly::from_nums(&gf, &[5, 5, 2, 2]));
}

#[test]
fn test_gf_poly_subtract2(){
    let gf: GF = GF::new(6, 0b1000011);
    let a = GFPoly::from_nums(&gf,           &[3, 0, 8, 0, 2, 1]);
    let b = GFPoly::from_nums(&gf,   /* - */ &[3, 0, 0, 1, 1, 1, 8, 8]);
    assert_eq!(a - b, GFPoly::from_nums(&gf, &[0, 0, 8, 1, 3, 0, 8, 8]));
}

#[test]
fn test_gf_poly_mult_by_scalar(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf,         /* 3 × */ &[0, 0, 2, 4]);
    assert_eq!(a * gf.num(3), GFPoly::from_nums(&gf, &[0, 0, 6, 12]));
}

#[test]
fn test_gf_poly_mult_by_scalar2(){
    let gf: GF = GF::new(6, 0b1000011);
    let a = GFPoly::from_nums(&gf,         /* 6 × */ &[3, 0, 1]);
    assert_eq!(a * gf.num(6), GFPoly::from_nums(&gf, &[10, 0, 6]));
}

#[test]
fn test_gf_poly_mult_by_zero(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[0, 3, 1]);
    assert_eq!(a * gf.num(0), GFPoly::zero(&gf));
}

#[test]
fn test_gf_poly_mult_by_poly(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[0, 1, 1]);
    let b = GFPoly::from_nums(&gf, &[3, 4]);
    assert_eq!(a * b, GFPoly::from_nums(&gf, &[0, 3, 7, 4]));
}

#[test]
fn test_gf_poly_mult_by_poly2(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[1, 2]);
    let b = GFPoly::from_nums(&gf, &[4, 3, 1]);
    assert_eq!(a * b, GFPoly::from_nums(&gf, &[4, 11, 7, 2]));
}

#[test]
fn test_gf_poly_mult_by_poly3(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[1, 1]); // 1 + X
    let b = GFPoly::from_nums(&gf, &[3, 1]); // 3 + X
    // (1 + X)(3 + X) = 3 + X + 3X + X²
    assert_eq!(a * b, GFPoly::from_nums(&gf, &[3, 2, 1]));
}

#[test]
fn test_gf_poly_mult_by_monomial(){
    let gf: GF = GF::new(4, 0b10011);
    let mut a = GFPoly::from_nums(&gf, &[0, 5, 4]);
    a <<= 3;
    assert_eq!(a, GFPoly::from_nums(&gf, &[0, 0, 0, 0, 5, 4]));
}

#[test]
fn test_gf_poly_mult_by_poly4(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[6, 1]);
    let b = GFPoly::from_nums(&gf, &[0, 1]);
    assert_eq!(a * b, GFPoly::from_nums(&gf, &[0, 6, 1]));
}

#[test]
fn test_gf_poly_rem(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[12, 26, 6]);
    let b = GFPoly::from_nums(&gf, &[4, 1]);
    assert_eq!(a % &b, GFPoly::from_nums(&gf, &[8]));
}

#[test]
fn test_gf_poly_rem2(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[2, 3, 3, 6, 9, 4]);
    let b = GFPoly::from_nums(&gf, &[9, 7, 0, 1]);
    assert_eq!(a % &b, GFPoly::from_nums(&gf, &[15, 4, 11]));
}

#[test]
fn test_gf_poly_rem3(){
    let gf: GF = GF::new(4, 0b10011);
    let a = GFPoly::from_nums(&gf, &[4, 3, 3, 7, 1, 2, 6, 1]);
    let b = GFPoly::from_nums(&gf, &[1, 0, 0, 3, 1]);
    assert_eq!(a % &b, GFPoly::from_nums(&gf, &[1, 14, 6, 9]));
}
