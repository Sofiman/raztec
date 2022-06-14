#![allow(dead_code)]
use std::{ops::{Add, Sub, Mul, Div}, fmt::{Display, Debug}};
use super::{poly::Polynomial, gf_poly::GFPoly};

pub struct GF {
    order: u8,
    size: usize,
    primitive: usize,
    exp_table: Vec<usize>,
    log_table: Vec<usize>
}

impl GF {
    pub fn new(n: u8, primitive: usize) -> Self {
        /* The order of the Galois Field is 2^n */
        let full_size = 2usize.pow(n as u32);
        let size = full_size - 1;
        let mut exp_table = vec![0; full_size];
        let mut log_table = vec![0; full_size];

        let mut x = 1;
        for (i, val) in exp_table.iter_mut().enumerate() {
            *val = x;
            log_table[x] = i;
            x *= 2;
            if x > size {
                x ^= primitive;
                x &= size;
            }
        }

        GF { order: n, size, primitive, exp_table, log_table }
    }

    pub fn num(&self, val: usize) -> GFNum {
        GFNum { gf: self, val: val % (self.size + 1) }
    }

    pub fn num_from_poly(&self, poly: Polynomial) -> GFNum {
        let val = poly.iter()
            .map(|&x| {
                if x != 0 && x != 1 {
                    None
                } else {
                    Some(x as usize)
                }
            })
            .map(|x| x.expect("Input Polynomial coefficients must be 0 or 1"))
            .fold(0, |acc, x| (acc << 1) + x);
        GFNum { val, gf: self }
    }

    pub fn to_gf_poly(&self, poly: Polynomial) -> GFPoly {
        let coeffs: Vec<GFNum> = poly.iter()
            .map(|&x| self.num(x as usize)).collect();
        GFPoly::new(self, &coeffs)
    }

    pub fn size(&self) -> usize {
        self.size + 1
    }

    pub fn order(&self) -> u8 {
        self.order
    }

    pub fn get_poly(&self, x: usize) -> Polynomial {
        let x = x as isize;
        let coeffs: Vec<isize> =
            (0..(self.order as isize)).map(|p| (x >> p) & 1).collect();
        Polynomial::new(&coeffs)
    }

    pub fn add(&self, a: GFNum, b: GFNum) -> GFNum {
        GFNum { val: a.val ^ b.val, gf: self }
    }

    pub fn sub(&self, a: GFNum, b: GFNum) -> GFNum {
        GFNum { val: a.val ^ b.val, gf: self }
    }

    pub fn mul(&self, a: GFNum, b: GFNum) -> GFNum {
        if a.val == 0 || b.val == 0 {
            return GFNum { val: 0, gf: self }
        }
        let x = self.log_table[a.val];
        let y = self.log_table[b.val];
        let val = self.exp_table[(x + y) % self.size];
        GFNum { val, gf: self }
    }

    pub fn div(&self, a: GFNum, b: GFNum) -> GFNum {
        self.mul(a, self.inv(b))
    }

    pub fn log2(&self, x: GFNum) -> GFNum {
        GFNum { val: self.log_table[x.val], gf: self }
    }

    pub fn exp2(&self, x: GFNum) -> GFNum {
        GFNum { val: self.exp_table[x.val], gf: self }
    }

    pub fn inv(&self, x: GFNum) -> GFNum {
        let i = self.size - self.log_table[x.val];
        GFNum { val: self.exp_table[i], gf: self }
    }

    /// Ordinary multiplication in GF(2^m)
    pub fn rep_add<'a>(&'a self, k: usize, x: GFNum<'a>) -> GFNum<'a> {
        if k % 2 == 0 {
            GFNum { val: 0, gf: self }
        } else {
            x
        }
    }
}

impl PartialEq for GF {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl Display for GF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF(2^{}, primitive: {})",
            self.order, self.get_poly(self.primitive))
    }
}

impl Debug for GF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF(2^{}, primitive: {})",
            self.order, self.get_poly(self.primitive))
    }
}

#[derive(Copy, PartialEq)]
pub struct GFNum<'a> {
    gf: &'a GF,
    val: usize
}

impl<'a> GFNum<'a> {
    pub fn value(&self) -> usize {
        self.val
    }

    pub fn group(&'a self) -> &'a GF {
        self.gf
    }

    pub fn as_poly(&self) -> Polynomial {
        self.gf.get_poly(self.val)
    }

    pub fn inv(self) -> GFNum<'a> {
        self.gf.inv(self)
    }

    pub fn zero(gf: &'a GF) -> GFNum<'a> {
        GFNum { gf, val: 0 }
    }
}

impl<'a> Clone for GFNum<'a> {
    fn clone(&self) -> Self {
        GFNum { gf: self.gf, val: self.val }
    }
}

impl<'a> Add for GFNum<'a> {
    type Output = GFNum<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.gf.order, rhs.gf.order, "Attempt to use finite field
            arithmetic on values from different fields");
        self.gf.add(self, rhs)
    }
}

impl<'a> Sub for GFNum<'a> {
    type Output = GFNum<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.gf.order, rhs.gf.order, "Attempt to use finite field
            arithmetic on values from different fields");
        self.gf.sub(self, rhs)
    }
}

impl<'a> Mul for GFNum<'a> {
    type Output = GFNum<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.gf.order, rhs.gf.order, "Attempt to use finite field
            arithmetic on values from different fields");
        self.gf.mul(self, rhs)
    }
}

impl<'a> Div for GFNum<'a> {
    type Output = GFNum<'a>;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.gf.order, rhs.gf.order, "Attempt to use finite field
            arithmetic on values from different fields");
        self.gf.div(self, rhs)
    }
}

impl Display for GFNum<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl Debug for GFNum<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
