use std::{ops::{Add, Sub, Mul, Div, Shl, Index}, fmt::Display, isize};
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
        let deg_a = self.deg();
        if deg_a < 0 {
            return rhs;
        }
        let deg_b = rhs.deg();
        if deg_b < 0 {
            return self;
        }

        let deg_a = deg_a as usize + 1;
        let deg_b = deg_b as usize + 1;
        let coeffs = if deg_a >= deg_b {
            self.coeffs.iter().take(deg_a)
                .zip(rhs.coeffs.iter().take(deg_b)
                    .chain(std::iter::repeat(&self.zero)))
                .map(|(&a, &b)| a + b).collect()
        }  else {
            rhs.coeffs.iter().take(deg_b)
                .zip(self.coeffs.iter().take(deg_a)
                    .chain(std::iter::repeat(&self.zero)))
                .map(|(&a, &b)| a + b).collect()
        };
        GFPoly { zero: self.zero, coeffs }
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
    type Output = GFPoly<'a>;

    fn mul(self, rhs: GFNum<'a>) -> Self::Output {
        let deg_a = self.deg();
        if deg_a < 0 || rhs == self.zero {
            return GFPoly { zero: self.zero, coeffs: Vec::new() }
        }
        let coeffs = self.coeffs.into_iter().take(deg_a as usize + 1)
            .map(|coef| coef * rhs).collect();
        GFPoly { zero: self.zero, coeffs }
    }
}

impl<'a> Shl<usize> for GFPoly<'a> {
    type Output = GFPoly<'a>;

    fn shl(self, rhs: usize) -> Self::Output {
        // mutiply by x^rhs
        let coeffs = std::iter::repeat(self.zero).take(rhs)
            .chain(self.coeffs.into_iter()).collect();
        GFPoly { zero: self.zero, coeffs }
    }
}

impl<'a> Div for GFPoly<'a> {
    type Output = (GFPoly<'a>, GFPoly<'a>);

    fn div(self, rhs: Self) -> Self::Output {
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
            panic!("The degree of the dividend must be greater or equal to the degree of the divisor");
        }
        let mut q = GFPoly {
            zero: self.zero,
            coeffs: vec![self.zero; deg_r as usize + 1]
        };
        let mut r = self;

        while deg_r >= deg_d {
            let lead = (deg_r - deg_d) as usize;
            let coef = r.coeffs[deg_r as usize] / rhs.coeffs[deg_d as usize];
            let divisor = (rhs.clone() * coef) << lead;
            q.coeffs[lead] = coef;
            r = r - divisor;
            deg_r = r.deg();
        }

        (q, r)
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
