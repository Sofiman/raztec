#![allow(dead_code)]
use std::{ops::{Add, Sub, Mul, Div, Index, Rem, MulAssign, ShlAssign, AddAssign},
fmt::{Display, Debug}, isize, vec::IntoIter};
use super::gf::*;

#[derive(Clone)]
pub struct GFPoly<'a> {
    // index corresponds to the power of x
    // example: coeffs[0] is equal to b in ax + b
    //          coeffs[1] is equal to a in ax + b
    coeffs: Vec<GFNum<'a>>,
    zero: GFNum<'a>
}

impl<'a> GFPoly<'a> {
    pub fn new(gf: &'a GF, coeffs: &[GFNum<'a>]) -> Self {
        Self { zero: gf.num(0), coeffs: coeffs.to_vec() }
    }

    pub fn from_nums(gf: &'a GF, coeffs: &[usize]) -> Self {
        let coeffs: Vec<GFNum> = coeffs.iter()
            .map(|&x| gf.num(x)).collect();
        Self { zero: gf.num(0), coeffs }
    }

    pub fn zero(gf: &'a GF) -> Self {
        Self { zero: gf.num(0), coeffs: vec![] }
    }

    pub fn deg(&self) -> isize {
        let mut i = self.coeffs.len();
        while i > 0 && self.coeffs[i - 1] == self.zero {
            i -= 1;
        }
        if i == 0 {
            isize::MIN
        } else {
            i as isize - 1
        }
    }

    pub fn iter(&'a self) -> std::slice::Iter<'a, GFNum<'a>> {
        self.coeffs.iter()
    }

    pub fn into_coeffs(self) -> IntoIter<GFNum<'a>> {
        self.coeffs.into_iter()
    }

    pub fn eval(&'a self, x: GFNum<'a>) -> GFNum<'a> {
        let mut p = x;
        let mut result = self.zero;
        for &coef in self.coeffs.iter() {
            result = result + p * coef;
            p = p * x;
        }
        result
    }
}

impl<'a> Index<usize> for GFPoly<'a> {
    type Output = GFNum<'a>;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.coeffs.len() {
            &self.zero
        } else {
            &self.coeffs[index]
        }
    }
}

impl PartialEq for GFPoly<'_> {
    fn eq(&self, other: &Self) -> bool {
        let deg = self.deg();
        let other_deg = other.deg();
        if deg != other_deg {
            return false;
        }
        if deg < 0 && other_deg < 0 {
            return true;
        }
        for i in 0..=(deg as usize) {
            if other.coeffs[i] != self.coeffs[i] {
                return false;
            }
        }
        true
    }
}

impl<'a> Add for GFPoly<'a> {
    type Output = GFPoly<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let deg_b = rhs.deg();
        if deg_b < 0 {
            return self;
        }
        let deg_a = self.deg();
        if deg_a < 0 {
            return rhs;
        }

        let deg_a = deg_a as usize + 1;
        let deg_b = deg_b as usize + 1;
        let coeffs = if deg_a >= deg_b {
            let mut out = self.coeffs;
            for (i, e) in out.iter_mut().take(deg_b).enumerate() {
                *e = *e + rhs.coeffs[i];
            }
            out
        }  else {
            let mut out = rhs.coeffs;
            for (i, e) in out.iter_mut().take(deg_a).enumerate() {
                *e = *e + self.coeffs[i];
            }
            out
        };
        GFPoly { zero: self.zero, coeffs }
    }
}

impl<'a> AddAssign<&GFPoly<'a>> for GFPoly<'a> {
    fn add_assign(&mut self, rhs: &Self) {
        let deg_b = rhs.deg();
        if deg_b < 0 {
            return;
        }
        let deg_a = self.deg();
        if deg_a < 0 {
            self.coeffs.clone_from(&rhs.coeffs);
            return;
        }

        let deg_a = deg_a as usize + 1;
        let deg_b = deg_b as usize + 1;
        if deg_a < deg_b {
            self.coeffs.extend_from_slice(&rhs.coeffs[deg_a..deg_b]);
        }
        for (i, e) in self.coeffs.iter_mut().take(deg_b).enumerate() {
            *e = *e + rhs.coeffs[i];
        }
    }
}

impl<'a> Sub for GFPoly<'a> {
    type Output = GFPoly<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.add(rhs) // In finite field arithmetic, add and sub are the same
    }
}

impl<'a> Mul for GFPoly<'a> {
    type Output = GFPoly<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let deg_a = self.deg();
        let deg_b = rhs.deg();
        if deg_a < 0 || deg_b < 0 {
            return GFPoly { zero: self.zero, coeffs: Vec::new() }
        }
        let deg_a = deg_a as usize;
        let deg_b = deg_b as usize;
        let mut out = vec![self.zero; deg_a + deg_b + 1];

        for i in 0..=deg_a {
            for j in 0..=deg_b {
                out[i + j] = out[i + j] + self.coeffs[i] * rhs.coeffs[j];
            }
        }

        GFPoly { zero: self.zero, coeffs: out }
    }
}

impl<'a> Mul<GFNum<'a>> for GFPoly<'a> {
    type Output = Self;

    fn mul(mut self, rhs: GFNum<'a>) -> Self {
        self *= rhs;
        self
    }
}

impl<'a> MulAssign<GFNum<'a>> for GFPoly<'a> {

    fn mul_assign(&mut self, rhs: GFNum<'a>) {
        if rhs == self.zero || self.deg() < 0 {
            self.coeffs.clear();
            return;
        }
        for coef in self.coeffs.iter_mut() {
            *coef = *coef * rhs;
        }
    }
}

impl<'a> ShlAssign<usize> for GFPoly<'a> {
    fn shl_assign(&mut self, rhs: usize) {
        // mutiply by x^rhs
        self.coeffs.splice(0..0, std::iter::repeat(self.zero).take(rhs));
    }
}

impl<'a> Div<&GFPoly<'a>> for GFPoly<'a> {
    type Output = (GFPoly<'a>, GFPoly<'a>);

    fn div(self, rhs: &Self) -> Self::Output {
        let deg_d = rhs.deg();
        if deg_d < 0 {
            panic!("GFPoly division by zero");
        }
        let mut deg_r = self.deg();
        if deg_r < 0 {
            let zero1 = GFPoly { zero: self.zero, coeffs: Vec::new() };
            let zero2 = GFPoly { zero: self.zero, coeffs: Vec::new() };
            return (zero1, zero2);
        }
        if deg_r < deg_d {
            panic!("The degree of the dividend must be greater or equal
                to the degree of the divisor");
        }
        let mut q = GFPoly {
            zero: self.zero,
            coeffs: vec![self.zero; deg_r as usize + 1]
        };
        let mut r = self;
        let mut divisor = rhs.clone();

        while deg_r >= deg_d {
            let lead = (deg_r - deg_d) as usize;
            let coef = r.coeffs[deg_r as usize] / rhs.coeffs[deg_d as usize];
            q.coeffs[lead] = coef;
            divisor *= coef;
            divisor <<= lead;
            r += &divisor; // here, it is actually a substraction
            deg_r = r.deg();
            divisor.clone_from(rhs); // set divisor back to rhs
        }

        (q, r)
    }
}

impl<'a> Rem<&GFPoly<'a>> for GFPoly<'a> {
    type Output = GFPoly<'a>;

    fn rem(self, rhs: &Self) -> Self::Output {
        let deg_d = rhs.deg();
        if deg_d < 0 {
            panic!("GFPoly division by zero");
        }
        let mut deg_r = self.deg();
        if deg_r < 0 {
            return GFPoly { zero: self.zero, coeffs: Vec::new() };
        }
        if deg_r < deg_d {
            panic!("The degree of the dividend must be greater or equal
                to the degree of the divisor");
        }
        let mut r = self;
        let mut divisor = rhs.clone();

        while deg_r >= deg_d {
            let lead = (deg_r - deg_d) as usize;
            divisor *= r.coeffs[deg_r as usize] / rhs.coeffs[deg_d as usize];
            divisor <<= lead;
            // this shift is unecessary in some cases, consider optimizing it
            r += &divisor; // here, it is actually a substraction
            deg_r = r.deg();
            divisor.clone_from(rhs); // set divisor back to rhs
        }

        r
    }
}

impl Display for GFPoly<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.deg() < 0 {
            return write!(f, "0");
        }
        for (power, &gfnum) in self.coeffs.iter().skip(1).enumerate().rev() {
            let coef = gfnum.value();
            if coef == 0 {
                continue;
            }
            if coef != 1 {
                write!(f, "{}", coef)?;
            }
            write!(f, "X")?;
            if power > 0 {
                write!(f, "^{}", power + 1)?;
            }
            write!(f, " + ")?;
        }
        write!(f, "{}", self.coeffs[0].value())?;
        Ok(())
    }
}

impl Debug for GFPoly<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.deg() < 0 {
            return write!(f, "0 over {}", self.zero.group());
        }
        for (power, &gfnum) in self.coeffs.iter().skip(1).enumerate().rev() {
            let coef = gfnum.value();
            if coef == 0 {
                continue;
            }
            if coef != 1 {
                write!(f, "{}", coef)?;
            }
            write!(f, "X")?;
            if power > 0 {
                write!(f, "^{}", power + 1)?;
            }
            write!(f, " + ")?;
        }
        write!(f, "{} over {}", self.coeffs[0].value(), self.zero.group())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
