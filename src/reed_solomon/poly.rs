use std::{ops::{Add, Sub, Mul, Div, Shl, Index, Rem}, fmt::Display, isize};

#[derive(Debug)]
pub struct Polynomial {
    // index corresponds to the power of x
    // example: coeffs[0] is equal to b in ax + b
    //          coeffs[1] is equal to a in ax + b
    coeffs: Vec<isize>
}

impl Polynomial {
    pub fn new(coeffs: &[isize]) -> Self {
        Self { coeffs: coeffs.to_vec() }
    }

    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    pub fn deg(&self) -> isize {
        if self.coeffs.is_empty() {
            return isize::MIN;
        }
        let mut i = self.coeffs.len() as isize - 1;
        while i >= 0 && self.coeffs[i as usize] == 0 {
            i -= 1;
        }
        if i == -1 {
            isize::MIN
        } else {
            i as isize
        }
    }

    pub fn iter(&self) -> std::slice::Iter<isize> {
        self.coeffs.iter()
    }

    pub fn eval(&self, x: isize) -> isize {
        let mut p = x;
        let mut result = 0;
        for coef in self.coeffs.iter() {
            result += p * coef;
            p *= x;
        }
        result
    }
}

impl Index<usize> for Polynomial {
    type Output = isize;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.coeffs.len() {
            &0
        } else {
            &self.coeffs[index]
        }
    }
}

impl Clone for Polynomial {
    fn clone(&self) -> Self {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        coeffs.extend(&self.coeffs);
        Polynomial { coeffs }
    }
}

impl PartialEq for Polynomial {
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

impl Add for Polynomial {
    type Output = Polynomial;

    fn add(self, rhs: Self) -> Self::Output {
        let deg_a = self.coeffs.len();
        let deg_b = rhs.coeffs.len();
        let max = if deg_a > deg_b { deg_a } else { deg_b };
        let mut out = vec![0; max];

        for i in 0..max {
            out[i] = self[i] + rhs[i];
        }

        Polynomial { coeffs: out }
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;

    fn sub(self, rhs: Self) -> Self::Output {
        let deg_a = self.coeffs.len();
        let deg_b = rhs.coeffs.len();
        let max = if deg_a > deg_b { deg_a } else { deg_b };
        let mut out = vec![0; max];

        for i in 0..max {
            out[i] = self[i] - rhs[i];
        }

        Polynomial { coeffs: out }
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: Self) -> Self::Output {
        let deg_a = self.deg();
        let deg_b = rhs.deg();
        if deg_a < 0 || deg_b < 0 {
            return Polynomial::zero()
        }
        let deg_a = deg_a as usize;
        let deg_b = deg_b as usize;
        let mut out = vec![0; deg_a + deg_b + 1];

        for i in 0..=deg_a {
            for j in 0..=deg_b {
                out[i + j] += self.coeffs[i] * rhs.coeffs[j];
            }
        }

        Polynomial { coeffs: out }
    }
}

impl Mul<isize> for Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: isize) -> Self::Output {
        let deg_a = self.deg();
        if deg_a < 0 || rhs == 0 {
            return Polynomial::zero()
        }
        let mut out = vec![0; self.coeffs.len()];

        for (i, coef) in self.coeffs.iter().enumerate() {
            out[i] = coef * rhs;
        }

        Polynomial { coeffs: out }
    }
}

impl Shl<usize> for Polynomial {
    type Output = Polynomial;

    fn shl(self, rhs: usize) -> Self::Output {
        // mutiply by x^rhs
        let mut coeffs = vec![0; rhs];
        coeffs.extend(&self.coeffs);
        Polynomial { coeffs }
    }
}

impl Div for Polynomial {
    type Output = (Polynomial, Polynomial);

    fn div(self, rhs: Self) -> Self::Output {
        let deg_d = rhs.deg();
        if deg_d < 0 {
            panic!("Polynomial division by zero");
        }
        let mut deg_r = self.deg();
        if deg_r < 0 {
            return (Polynomial::zero(), Polynomial::zero());
        }
        if deg_r < deg_d {
            panic!("The degree of the dividend must be greater or equal to the degree of the divisor");
        }
        let mut q = Polynomial { coeffs: vec![0; self.coeffs.len()] };
        let mut r = self;
        let d = q.clone() + rhs;

        while deg_r >= deg_d {
            let lead = (deg_r - deg_d) as usize;
            let coef = r[deg_r as usize];
            let divisor = (d.clone() << lead) * coef;
            q.coeffs[lead] += coef;
            r = r - divisor.clone();
            deg_r = r.deg();
        }

        (q, r)
    }
}

impl Rem<usize> for Polynomial {
    type Output = Polynomial;

    fn rem(self, rhs: usize) -> Self::Output {
        let coeffs = self.coeffs.to_vec()
            .iter().map(|&x| x.rem_euclid(rhs as isize))
            .collect();
        Polynomial { coeffs }
    }
}

impl Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.deg() < 0 {
            return write!(f, "0");
        }
        for (power, &coef) in self.coeffs.iter().skip(1).enumerate().rev() {
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
        write!(f, "{}", self.coeffs[0])?;
        Ok(())
    }
}
