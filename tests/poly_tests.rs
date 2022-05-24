use raztec::reed_solomon::poly::*;

#[test]
fn test_poly_deg_zero(){
    assert_eq!(Polynomial::zero().deg(), isize::MIN);
}

#[test]
fn test_poly_deg_simple(){
    assert_eq!(Polynomial::new(&[3, 3]).deg(), 1);
}

#[test]
fn test_poly_deg_simple2(){
    assert_eq!(Polynomial::new(&[7]).deg(), 0);
}

#[test]
fn test_poly_deg_padding_zeros(){
    assert_eq!(Polynomial::new(&[1, 0]).deg(), 0);
}

#[test]
fn test_poly_deg_normal(){
    assert_eq!(Polynomial::new(&[1, 0, 5, 0, 3]).deg(), 4);
}

#[test]
fn test_poly_index(){
    assert_eq!(Polynomial::new(&[7, 8, 2, 3])[2], 2);
}

#[test]
fn test_poly_add_zero(){
    let a = Polynomial::new(&[1, 2]);
    assert_eq!(a.clone() + Polynomial::zero(), a);
}

#[test]
fn test_poly_add(){
    let a = Polynomial::new(&[3, -1, 2, 6]);
    let b = Polynomial::new(&[0, 1, 4]);
    assert_eq!(a + b, Polynomial::new(&[3, 0, 6, 6]));
}

#[test]
fn test_poly_add_opposite(){
    let a = Polynomial::new(&[2, 1, 2]);
    let b = Polynomial::new(&[-2, -1, -2]);
    assert_eq!(a + b, Polynomial::zero());
}

#[test]
fn test_poly_subtract_zero(){
    let a = Polynomial::new(&[4, 5, 1]);
    assert_eq!(a.clone() - Polynomial::zero(), a);
}

#[test]
fn test_poly_subtract(){
    let a = Polynomial::new(&[9, 0, 1, 2]);
    let b = Polynomial::new(&[0, 5, 3]);
    assert_eq!(a - b, Polynomial::new(&[9, -5, -2, 2]));
}

#[test]
fn test_poly_zero_subtract(){
    let a = Polynomial::new(&[5, 6, 6]);
    assert_eq!(Polynomial::zero() - a, Polynomial::new(&[-5, -6, -6]));
}

#[test]
fn test_poly_mult_by_scalar(){
    let a = Polynomial::new(&[0, 0, 2, -4]);
    assert_eq!(a * -3, Polynomial::new(&[0, 0, -6, 12]));
}

#[test]
fn test_poly_mult_by_zero(){
    let a = Polynomial::new(&[0, 3, 1]);
    assert_eq!(a * Polynomial::zero(), Polynomial::zero());
}

#[test]
fn test_poly_mult_by_poly(){
    let a = Polynomial::new(&[0, 1, 1]);
    let b = Polynomial::new(&[3, 4]);
    assert_eq!(a * b, Polynomial::new(&[0, 3, 7, 4]));
}

#[test]
fn test_poly_mult_by_poly2(){
    let a = Polynomial::new(&[1, 2]);
    let b = Polynomial::new(&[4, 3, 1]);
    assert_eq!(a * b, Polynomial::new(&[4, 11, 7, 2]));
}

#[test]
fn test_poly_mult_by_poly3(){
    let a = Polynomial::new(&[-2, 1]);
    let b = Polynomial::new(&[3, 1]);
    assert_eq!(a * b, Polynomial::new(&[-6, 1, 1]));
}

#[test]
fn test_poly_mult_by_poly4(){
    let a = Polynomial::new(&[-2, 1]);
    let b = Polynomial::new(&[0, 1]);
    assert_eq!(a * b, Polynomial::new(&[0, -2, 1]));
}

#[test]
fn test_poly_division(){
    let a = Polynomial::new(&[-1, -5, 2]);
    let b = Polynomial::new(&[-3, 1]);
    assert_eq!(a / b, (Polynomial::new(&[1, 2]), Polynomial::new(&[2])));
}

#[test]
fn test_poly_division2(){
    let a = Polynomial::new(&[-9, 6, 0, 0, 2, 0, 1]);
    let b = Polynomial::new(&[3, 0, 0, 1]);
    assert_eq!(a / b, (Polynomial::new(&[-3, 2, 0, 1]), Polynomial::zero()));
}

#[test]
fn test_poly_division3(){
    let a = Polynomial::new(&[-17, 38, -12, 1, 0, 6]);
    let b = Polynomial::new(&[-3, 1, 1]);
    assert_eq!(a / b, (Polynomial::new(&[-55, 25, -6, 6]), Polynomial::new(&[-182, 168])));
}

#[test]
fn test_poly_mult_div_comp(){
    let a = Polynomial::new(&[9, 0]);
    let b = Polynomial::new(&[0, 0, 0, 0, 0, 1]);
    let c = Polynomial::new(&[1, 0, 6, 4, 11, 1]);
    assert_eq!(a * b / c, (Polynomial::new(&[9]), Polynomial::new(&[-9, 0, -54, -36, -99])));
}
