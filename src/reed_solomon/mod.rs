//! Reed Solomon Error Correction implementation
//!
//! This modules contain different structs to generate the necessary error
//! check codes and the reed solomon error correction algorithm.
//!
//! Note that this is my own implementation of the Reed Solomon and all the
//! maths required (Galois Field implementation) and is a bit slow generating
//! check codes with big data sets.

mod gf;
mod poly;
mod gf_poly;

use self::gf::*;
use self::gf_poly::GFPoly;

/// Reed Solomon error correction code generator
pub struct ReedSolomon {
    gf: GF
}

impl ReedSolomon {

    /// Create a new ReedSolomon in GF(2^m) with the specified primitive
    pub fn new(m: u8, primitive: usize) -> Self {
        ReedSolomon { gf: GF::new(m, primitive) }
    }

    /// Generates `k` Reed Solomon check codes using the `data` input. Returns
    /// an iterator over these check codes.
    ///
    /// # Arguments
    /// * `data` - The input data
    /// * `k` - Number of check codes to generate
    pub fn generate_check_codes(&self, data: &[usize], k: usize) ->
        impl Iterator<Item = usize> + '_ {
        let coeffs: Vec<GFNum> = data.iter()
            .map(|&x| self.gf.num(x)).rev().collect();
        let mut poly = GFPoly::new(&self.gf, &coeffs);
        poly <<= k;

        // Calculate the generator polynomial by calculating
        // the polynomial (x-2^1)(x-2^2)...(x-2^k)
        let mut coeffs = vec![self.gf.num(0); k + 1];
        coeffs[0] = self.gf.num(1);

        for i in 1..=k {
            coeffs[i] = coeffs[i - 1];
            let p = self.gf.exp2(self.gf.num(i));
            for j in (1..i).rev() {
                coeffs[j] = coeffs[j - 1] + (coeffs[j] * p);
            }
            coeffs[0] = coeffs[0] * p;
        }
        let generator = GFPoly::new(&self.gf, &coeffs);

        let rem = poly % &generator;
        rem.into_coeffs().take(k).rev().map(|x| x.value())
    }

    /// Decode the data payload and fix the possible errors in-place.
    ///
    /// # Arguments
    /// * `data` - The input payload (received data + received check codes)
    /// * `n` - The number of actual data packets in the data slice
    pub fn fix_errors(&self, data: &mut [usize], n: usize)
        -> Result<(), ()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_check_codes(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0, 0b1001], 5).collect();
        assert_eq!(check_codes, [12, 2, 3, 1, 9]);
    }

    #[test]
    fn test_generate_check_codes2(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0, 0b0100], 5).collect();
        assert_eq!(check_codes, [0b1010, 0b0011, 0b1011, 0b1000, 0b0100]);
    }

    #[test]
    fn test_generate_check_codes3(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0b0101, 0b1010], 5).collect();
        assert_eq!(check_codes, [0b1110, 0b0111, 0b0101, 0b0000, 0b1011]);
    }

    #[test]
    fn test_generate_check_codes4(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0b1111, 0b0000], 5).collect();
        assert_eq!(check_codes, [0b0111, 0b1000, 0b0111, 0b1001, 0b0011]);
    }

    #[test]
    fn test_generate_check_big(){
        let rs = ReedSolomon::new(6, 0b1000011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[39, 50, 1, 28, 7, 2, 42, 40, 37, 15], 7)
            .collect();
        assert_eq!(check_codes, [44, 29, 43, 52, 49, 22, 15]);
    }

    #[test]
    fn test_fix_errors(){
        let mut inp = [0b0101, 0b1101, 0b0111, 0b1001, 0b0101, 0b0000, 0b1011];
        let     exp = [0b0101, 0b1010, 0b1110, 0b0111, 0b0101, 0b0000, 0b1011];
        let rs = ReedSolomon::new(4, 0b10011);
        //rs.fix_errors(&mut inp, 2).unwrap();
        //assert_eq!(inp, exp);
    }
}
