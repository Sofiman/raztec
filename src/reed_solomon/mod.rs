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
    /// It uses the Euclidean Decoder to calculate the error locator and error
    /// magnitudes polynomials.
    ///
    /// # Arguments
    /// * `msg` - The input payload (received data + received check codes)
    /// * `k` - The number of check codes within the data slice
    pub fn fix_errors(&self, msg: &mut [usize], k: usize) -> Result<(), String> {
        let mut coeffs: Vec<GFNum> = msg.iter()
            .map(|&x| self.gf.num(x)).rev().collect();
        let msg_poly = GFPoly::from_vec(&self.gf, coeffs);
        let s = self.syndromes(&msg_poly, k);
        if s.deg() == isize::MIN { // S(X) = 0 <=> no errors
            return Ok(());
        }

        // Precompute x^k
        let (zero, one) = (self.gf.num(0), self.gf.num(1));
        coeffs = vec![zero; k + 1];
        coeffs[k] = one;
        let xk = GFPoly::from_vec(&self.gf, coeffs);

        // Apply the Extended Euclidean algorithm until r has a degree < t/2
        let (e_loc, e_mag) = Self::e_gcd(&self.gf, k as isize, &xk, s.clone());

        // Forney algorithm
        let e_eval = (s / &e_loc) % &xk;

        todo!()
    }

    /// Compute the syndromes polynomial. It is used to quickly check if the
    /// input message is corrupted or not.
    fn syndromes<'a>(&'a self, poly: &'a GFPoly, k: usize) -> GFPoly<'a> {
        let mut coeffs = vec![self.gf.num(0); k];
        for i in 1..=k {
            let p = self.gf.exp2(self.gf.num(i));
            coeffs[i - 1] = poly.eval(p);
        }
        GFPoly::from_vec(&self.gf, coeffs)
    }

    /// Extended Euclidean algorithm adapted by Sugiyama.
    /// Outputs (error locator, error magnitudes) polynomials.
    fn e_gcd<'a>(gf: &'a GF, mut t: isize, xk: &'a GFPoly, s: GFPoly<'a>)
        -> (GFPoly<'a>, GFPoly<'a>) {
        let (mut old_r, mut r) = (xk.clone(), s); // x^k
        let (mut old_a, mut a) = (GFPoly::zero(gf),GFPoly::from_nums(gf,&[1]));

        t /= 2;
        while r.deg() >= t {
            let q = old_r.clone() / &r;

            let r2 = old_r;
            old_r = r;
            r = r2 - q.clone() * old_r.clone();

            let s2 = old_a;
            old_a = a;
            a = s2 - q.clone() * old_a.clone();
        }

        // divide both a and r by the low order term of a to satisfy a[0] = 1
        let coef = a[0].inv();
        (a * coef, r * coef) // (error_locator, error_magnitudes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rs_generate_check_codes_zero(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0, 0], 3).collect();
        assert_eq!(check_codes, [0, 0, 0]);
    }

    #[test]
    fn test_rs_generate_check_codes(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0, 0b1001], 5).collect();
        assert_eq!(check_codes, [12, 2, 3, 1, 9]);
    }

    #[test]
    fn test_rs_generate_check_codes2(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0, 0b0100], 5).collect();
        assert_eq!(check_codes, [0b1010, 0b0011, 0b1011, 0b1000, 0b0100]);
    }

    #[test]
    fn test_rs_generate_check_codes3(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0b0101, 0b1010], 5).collect();
        assert_eq!(check_codes, [0b1110, 0b0111, 0b0101, 0b0000, 0b1011]);
    }

    #[test]
    fn test_rs_generate_check_codes4(){
        let rs = ReedSolomon::new(4, 0b10011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[0b1111, 0b0000], 5).collect();
        assert_eq!(check_codes, [0b0111, 0b1000, 0b0111, 0b1001, 0b0011]);
    }

    #[test]
    fn test_rs_generate_check_big(){
        let rs = ReedSolomon::new(6, 0b1000011);
        let check_codes: Vec<usize> = rs
            .generate_check_codes(&[39, 50, 1, 28, 7, 2, 42, 40, 37, 15], 7)
            .collect();
        assert_eq!(check_codes, [44, 29, 43, 52, 49, 22, 15]);
    }

    #[test]
    fn test_rs_syndromes_with_valid_code(){
        let inp: Vec<usize> = [0, 0b1001, 12, 2, 3, 1, 9].into_iter().rev()
            .collect();
        let rs = ReedSolomon::new(4, 0b10011);
        let inp_poly = GFPoly::from_nums(&rs.gf, &inp);
        assert_eq!(GFPoly::from_nums(&rs.gf, &[]), rs.syndromes(&inp_poly, 5));
    }

    #[test]
    fn test_rs_syndromes_with_corrupted_code(){
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
    fn test_rs_fix_errors(){
        let mut inp = [0b0101, 0b1101, 0b0111, 0b1001, 0b0101, 0b0000, 0b1011];
        let     exp = [0b0101, 0b1010, 0b0111, 0b0111, 0b0101, 0b0000, 0b1011];
        let rs = ReedSolomon::new(4, 0b10011);
        rs.fix_errors(&mut inp, 5).unwrap();
        assert_eq!(inp, exp);
    }
}
