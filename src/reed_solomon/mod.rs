pub mod gf;
pub mod poly;
pub mod gf_poly;

use self::gf::{GF, GFNum};
use self::gf_poly::GFPoly;

pub struct ReedSolomonEncoder {
    gf: GF
}

impl ReedSolomonEncoder {
    pub fn new(m: u8, primitive: usize) -> Self {
        ReedSolomonEncoder { gf: GF::new(m, primitive) }
    }

    pub fn generate_check_codes(&self, data: &[usize], k: usize) -> Vec<usize> {
        let coeffs: Vec<GFNum> = data.iter()
            .map(|&x| self.gf.num(x)).rev().collect();
        let poly = GFPoly::new(&self.gf, &coeffs) << k;

        let mut generator = GFPoly::new(&self.gf, &[self.gf.num(1)]);
        for i in 1..=k {
            let power = self.gf.exp2(self.gf.num(i));
            let next = GFPoly::new(&self.gf, &[power, self.gf.num(1)]);
            generator = generator * next;
        }

        let (_, rem) = poly / generator;
        rem.iter().take(k).rev().map(|&x| x.value()).collect()
    }
}
