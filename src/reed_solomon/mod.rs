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
        let zero = self.gf.num(0);
        let len = data.len();

        let mut coeffs = vec![zero; len + k];
        coeffs.splice(k.., data.iter().map(|&x| self.gf.num(x)).rev());
        let poly = GFPoly::new(&self.gf, &coeffs);

        // Calculate the generator polynomial by calculating
        // the polynomial (x-2^1)(x-2^2)...(x-2^k)

        // init
        coeffs.truncate(k + 1); // reuse the allocated coeffs array
        coeffs.fill(zero);
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

        // Finally, calculate the reminder to get the check codes
        let rem = poly % &generator;
        rem.into_coeffs().take(k).rev().map(|x| x.value())
    }

    /// Decode the data payload and fix the possible errors in-place.
    /// This is using the Gao decoder developed by Shuhong Gao.
    ///
    /// # Arguments
    /// * `msg` - The input payload (received data + received check codes)
    /// * `k` - The number of check codes within the data slice
    pub fn fix_errors(&self, msg: &mut [usize], k: usize) -> Result<(), String> {
        let mut coeffs: Vec<GFNum> = msg.iter()
            .map(|&x| self.gf.num(x)).rev().collect();
        let msg_poly = GFPoly::new(&self.gf, &coeffs);
        let g1 = self.syndromes(&msg_poly, k);
        if g1.deg() == isize::MIN { // s(X) = 0 <=> no errors
            return Ok(());
        }

        let zero = self.gf.num(0);
        coeffs.truncate(k + 1); // reuse the allocated coeffs array
        coeffs.fill(zero);
        coeffs[0] = self.gf.num(1);

        for i in 1..=k {
            coeffs[i] = coeffs[i - 1];
            let p = self.gf.exp2(self.gf.num(i));
            for j in (1..i).rev() {
                coeffs[j] = coeffs[j - 1] + (coeffs[j] * p);
            }
            coeffs[0] = coeffs[0] * p;
        }
        let g0 = GFPoly::new(&self.gf, &coeffs);

        let g = g0.gcd(&g1);
        let (_, v) = Self::bezout(&g, &g0, &g1, msg.len() / 2);
        let (f1, r) = g.clone() / &v;
        if f1.deg() < k as isize && r.deg() < 0 { // deg(f1) < k && r = 0
            for (i, x) in f1.iter().rev().enumerate() {
                msg[i] = x.value();
            }
            Ok(())
        } else {
            Err("Reed Solomon failed because of too many errors".to_owned())
        }
    }

    /// Compute the syndromes polynomial. It is used to quickly check if the
    /// input message is corrupted or not.
    fn syndromes<'a>(&'a self, poly: &'a GFPoly, k: usize) -> GFPoly<'a> {
        let mut coeffs = vec![self.gf.num(0); k];
        for i in 1..=k {
            let p = self.gf.exp2(self.gf.num(i));
            coeffs[i - 1] = poly.eval(p);
        }
        GFPoly::new(&self.gf, &coeffs)
    }

    /// Extended Euclidean algorithm
    fn bezout<'a>(gcd: &'a GFPoly, a: &'a GFPoly, b: &'a GFPoly, l: usize)
        -> (GFPoly<'a>, GFPoly<'a>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_check_codes_zero(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0, 0], 3).collect();
        assert_eq!(check_codes, [0, 0, 0]);
    }

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
    fn test_syndromes_with_valid_code(){
        let inp: Vec<usize> = [0, 0b1001, 12, 2, 3, 1, 9].into_iter().rev()
            .collect();
        let rs = ReedSolomon::new(4, 0b10011);
        let inp_poly = GFPoly::from_nums(&rs.gf, &inp);
        assert_eq!(GFPoly::from_nums(&rs.gf, &[]), rs.syndromes(&inp_poly, 5));
    }

    #[test]
    fn test_syndromes_with_corrupted_code(){
        // original: [15, 0, 7, 8, 7, 9, 3]
        // corrupted: [8, 10, 7, 8, 7, 9, 3]
        let inp = [0b1000, 0b1010, 0b0111, 0b1000, 0b0111, 0b1001, 0b0011]
            .into_iter().rev().collect::<Vec<usize>>();
        let exp = [5, 6, 13, 13, 11];
        let rs = ReedSolomon::new(4, 0b10011);
        let inp_poly = GFPoly::from_nums(&rs.gf, &inp);
        assert_eq!(GFPoly::from_nums(&rs.gf, &exp), rs.syndromes(&inp_poly, 5));
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
