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
    /// * `t` - The number of check codes within the data slice
    pub fn fix_errors(&self, msg: &mut [usize], t: usize) -> Result<(), String> {
        let mut coeffs: Vec<GFNum> = msg.iter()
            .map(|&x| self.gf.num(x)).rev().collect();
        let msg_poly = GFPoly::from_vec(&self.gf, coeffs);
        println!("R(x): {}", msg_poly);
        let s = self.syndromes(&msg_poly, t);
        if s.deg() == isize::MIN { // S(X) = 0 <=> no errors
            return Ok(());
        }
        println!("S(x): {}", s);

        // Precompute x^t
        coeffs = vec![self.gf.num(0); t + 1];
        coeffs[t] = self.gf.num(1);
        let xt = GFPoly::from_vec(&self.gf, coeffs);

        // Apply the Extended Euclidean algorithm until r has a degree < t/2
        let (mut lambda, omega) = Self::e_gcd(&self.gf, t as isize / 2, xt, s)?;
        println!("Λ(x) = {}", lambda);
        println!("Ω(x) = {}", omega);
        let err_locs = Self::find_error_roots(&self.gf, &lambda)?;
        println!("locations: {:?}", err_locs);
        let err_mags = Self::find_error_vals(&self.gf,
            &mut lambda, &omega, &err_locs);
        println!("magnitudes: {:?}", err_mags);

        let l = msg.len();
        for (&loc, &mag) in err_locs.iter().zip(err_mags.iter()) {
            let pos = self.gf.log2(loc).value() - 1;
            println!("(*) log({}) = {}", loc, pos);
            if pos > l {
                return Err("Invalid error locations".to_owned());
            }
            msg[l - 1 - pos] = (msg_poly[pos] + mag).value();
        }

        Ok(())
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
    fn e_gcd<'a>(gf: &'a GF, t: isize, xt: GFPoly<'a>, s: GFPoly<'a>)
        -> Result<(GFPoly<'a>, GFPoly<'a>), String> {
        let (mut old_r, mut r) = (xt, s); // x^k
        let (mut old_a, mut a) = (GFPoly::zero(gf),GFPoly::from_nums(gf,&[1]));

        while r.deg() >= t {
            let q = old_r.clone() / &r;

            let r2 = old_r;
            old_r = r;
            r = r2 - q.clone() * old_r.clone();

            let a2 = old_a;
            old_a = a;
            a = a2 - q.clone() * old_a.clone();
        }
        let coef = a[0];
        if coef.value() == 0 {
            return Err("EEA failed".to_owned());
        }

        // divide both a and r by the low order term of a to satisfy a[0] = 1
        let coef = coef.inv();
        Ok((a * coef, r * coef)) // (error_locator, error_magnitudes)
    }

    /// Chien's search algorithm
    fn find_error_roots<'a>(gf: &'a GF, e_loc: &GFPoly<'a>)
        -> Result<Vec<GFNum<'a>>, String> {
        let nb_err = e_loc.deg();
        if nb_err < 0 {
            return Err("Invalid number of errors".to_owned());
        }
        let nb_err = nb_err as usize;
        let mut roots = Vec::with_capacity(nb_err);
        let (mut i, l) = (1, gf.size());
        while i < l && roots.len() < nb_err {
            let n = gf.num(i);
            if e_loc.eval(n).value() == 0 {
                roots.push(n.inv());
            }
            i += 1;
        }
        if roots.len() != nb_err {
            Err("Invalid error locator (Too many errors?)".to_owned())
        } else {
            Ok(roots)
        }

    }

    /// Forney's algorithm
    fn find_error_vals<'a>(gf: &'a GF, sigma: &'a mut GFPoly<'a>,
        omega: &'a GFPoly<'a>, err_locs: &[GFNum<'a>]) -> Vec<GFNum<'a>> {
        // Calculate Λ'(x) in GF
        GFPoly::fm_derive(sigma, gf);

        err_locs.iter().map(|x| {
            let xi = x.inv();
            omega.eval(xi) / sigma.eval(xi)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, thread_rng};
    use rand::distributions::Uniform;

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
        let mut inp = [0b1000, 0b0000, 0b0111, 0b1000, 0b0111, 0b1001, 0b0011];
        let     exp = [0b1111, 0b0000, 0b0111, 0b1000, 0b0111, 0b1001, 0b0011];
        let rs = ReedSolomon::new(4, 0b10011);
        rs.fix_errors(&mut inp, 5).unwrap();
        assert_eq!(inp, exp);
    }

    #[test]
    fn test_rs_fix_errors2(){
        let mut inp = [0b1101, 0b1000, 0b1010, 0b0011, 0b1011, 0b1000, 0b0100];
        let     exp = [0b0000, 0b0100, 0b1010, 0b0011, 0b1011, 0b1000, 0b0100];
        let rs = ReedSolomon::new(4, 0b10011);
        rs.fix_errors(&mut inp, 5).unwrap();
        assert_eq!(inp, exp);
    }

    #[test]
    fn test_rs_fix_errors_big(){
        let rs = ReedSolomon::new(6, 0b1000011);
        let mut inp = [39, 50, 1, 28, 7, 2, 42, 30, 14, 25, 44, 29, 43, 52, 49, 22, 15];
        let     exp = [39, 50, 1, 28, 7, 2, 42, 40, 37, 15, 44, 29, 43, 52, 49, 22, 15];
        rs.fix_errors(&mut inp, 7).unwrap();
        assert_eq!(inp, exp);
    }

    #[test]
    fn test_rs_mixed(){
        let mut rng = thread_rng();
        let n = rng.gen_range(20..=40);
        let nb_err = rng.gen_range(2..=10);
        let mut data: Vec<usize> = (&mut rng).sample_iter(Uniform::new(0, 256))
            .take(n).collect();
        println!("Generated data: {:?}", data);

        let rs = ReedSolomon::new(8, 0x171);
        data.extend(rs.generate_check_codes(&data, nb_err*2));
        let original = data.clone();
        let l = data.len();

        println!("Corrupting {} values", nb_err);
        for _ in 0..nb_err {
            data[rng.gen_range(0..l)] = rng.gen_range(0..256);
        }
        println!("Corrupted data: {:?}", data);

        rs.fix_errors(&mut data, nb_err*2).unwrap();
        assert_eq!(data, original);
    }
}
