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
        let g0 = GFPoly::new(&self.gf, &coeffs); // generator polynomial

        // g = u(x) * g0(x) + v(x) * g1(x)
        let deg_limit = (self.gf.order() as usize + k) as isize / 2;
        let (g, _, v) = Self::p_bezout(&self.gf, &g0, &g1, deg_limit);
        let (f1, r) = g / &v;
        println!("=> {} | {}", f1, r);
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

    /// [Partial] Extended Euclidean algorithm. Outputs (g, u, v) to satisfy
    /// Bezout's identity: au + bv = g
    pub fn p_bezout<'a>(gf: &'a GF, a: &'a GFPoly, b: &'a GFPoly, limit: isize)
        -> (GFPoly<'a>, GFPoly<'a>, GFPoly<'a>) {
        let (mut r0, mut r1) = (a.clone(), b.clone());
        let (mut s0, mut s1) = (GFPoly::from_nums(gf, &[1]), GFPoly::zero(gf));

        while r1.deg() >= 0 {
            let (q, _) = r0.clone() / &r1;

            let r2 = r0;
            r0 = r1;
            r1 = r2 - q.clone() * r0.clone();

            let s2 = s0;
            s0 = s1;
            s1 = s2 - q.clone() * s0.clone();
        }

        let t = if b.deg() < 0 {
            GFPoly::zero(gf)
        } else {
            let (q, _) = (r0.clone() - s0.clone() * a.clone()) / b;
            q
        };

        (r0, s0, t)
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
    fn test_rs_p_bezout(){
        let rs = ReedSolomon::new(4, 0b10011);
        let a = GFPoly::from_nums(&rs.gf, &[1, 3, 5, 6, 1]);
        let b = GFPoly::from_nums(&rs.gf, &[1, 4, 0, 1, 8]);
        let (d, u, v) = ReedSolomon::p_bezout(&rs.gf, &a, &b, 0);
        assert_eq!(a.clone() * u + b.clone() * v, d);
        assert_eq!(a.gcd(&b), d);
    }

    #[test]
    fn test_rs_p_bezout2(){
        let rs = ReedSolomon::new(8, 0x171);
        let a = GFPoly::from_nums(&rs.gf, &[11, 3, 8, 7, 15, 1]);
        let b = GFPoly::from_nums(&rs.gf, &[2, 1, 0, 5, 0, 1, 7, 1, 1, 1]);
        let exp =  (GFPoly::from_nums(&rs.gf, &[29]),
                    GFPoly::from_nums(&rs.gf, &[167, 120, 116, 115, 186, 81, 231, 228, 121]),
                    GFPoly::from_nums(&rs.gf, &[132, 219, 0, 216, 121]));
        let (d, u, v) = ReedSolomon::p_bezout(&rs.gf, &a, &b, 0);
        assert_eq!(a.clone() * u.clone() + b.clone() * v.clone(), d);
        assert_eq!((d, u, v), exp);
    }

    #[test]
    fn test_rs_p_bezout3(){
        // https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Example_2
        let rs = ReedSolomon::new(8, 0x171);
        let a = GFPoly::from_nums(&rs.gf, &[1, 1, 0, 1, 1, 0, 0, 0, 1]);
        let b = GFPoly::from_nums(&rs.gf, &[1, 1, 0, 0, 1, 0, 1]);
        let exp =  (GFPoly::from_nums(&rs.gf, &[1]),
                    GFPoly::from_nums(&rs.gf, &[1, 0, 1, 1, 1, 1]),
                    GFPoly::from_nums(&rs.gf, &[0, 1, 0, 1, 0, 0, 1, 1]));
        assert_eq!(ReedSolomon::p_bezout(&rs.gf, &a, &b, 0), exp);
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
