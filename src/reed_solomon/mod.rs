pub mod gf;
pub mod poly;
pub mod gf_poly;

use self::poly::Polynomial;
use self::gf::{GF, GFNum};
use self::gf_poly::GFPoly;

pub struct ReedSolomonEncoder {
    gf: GF,
    generator: Polynomial
}

impl ReedSolomonEncoder {
    pub fn new(m: u8, primitive: usize, generator: Polynomial) -> Self {
        ReedSolomonEncoder { gf: GF::new(m, primitive), generator }
    }

    pub fn generate_check_codes(&self, data: &[u8], k: usize) -> Vec<u8> {
        let coeffs: Vec<GFNum> = data.iter()
            .map(|&x| self.gf.num(x as usize)).rev().collect();
        let poly = GFPoly::new(&self.gf, &coeffs) << k;
        let gen = self.gf.to_gf_poly(self.generator.clone());
        let (_, rem) = poly / gen;
        rem.iter().take(k).rev().map(|&x| x.value() as u8).collect()
    }
}
