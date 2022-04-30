use std::{ops::{Add, Sub, Mul, Div}, fmt::Display};

use super::poly::Polynomial;

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

        let mut x = 1;
        for val in exp_table.iter_mut() {
            *val = x;
            x *= 2;
            if x > size {
                x ^= primitive;
                x &= size;
            }
        }
        println!("{:?}", exp_table);

        let mut log_table = vec![0; full_size];
        for i in 0..size {
            log_table[exp_table[i]] = i;
        }
        println!("{:?}", log_table);

        GF { order: n, size, primitive, exp_table, log_table }
    }

    pub fn num(&self, val: usize) -> GFNum {
        GFNum { gf: self, val: val % self.size }
    }

    pub fn poly(&self, poly: Polynomial) -> GFNum {
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

    pub fn size(&self) -> usize {
        self.size + 1
    }

    pub fn order(&self) -> u8 {
        self.order
    }

    pub fn as_poly(&self, x: usize) -> Polynomial {
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
}

impl Display for GF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF(size: {}, primitive: {})", 
            self.size, self.as_poly(self.primitive))
    }
}

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
        self.gf.as_poly(self.val)
    }

    pub fn inv(self) -> GFNum<'a> {
        self.gf.inv(self)
    }
}

impl<'a> Add for GFNum<'a> {
    type Output = GFNum<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.gf.order != rhs.gf.order {
            panic!("Attempt to use finite field arithmetic on values from different fields");
        }
        self.gf.add(self, rhs)
    }
}

impl<'a> Sub for GFNum<'a> {
    type Output = GFNum<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.gf.order != rhs.gf.order {
            panic!("Attempt to use finite field arithmetic on values from different fields");
        }
        self.gf.sub(self, rhs)
    }
}

impl<'a> Mul for GFNum<'a> {
    type Output = GFNum<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.gf.order != rhs.gf.order {
            panic!("Attempt to use finite field arithmetic on values from different fields");
        }
        self.gf.mul(self, rhs)
    }
}

impl<'a> Div for GFNum<'a> {
    type Output = GFNum<'a>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.gf.order != rhs.gf.order {
            panic!("Attempt to use finite field arithmetic on values from different fields");
        }
        self.gf.div(self, rhs)
    }
}
