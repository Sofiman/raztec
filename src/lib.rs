//! # raztec
//!
//! A library for reading and writing Aztec 2D Barcodes using no external lib

pub mod writer;
pub mod reader;
pub mod reed_solomon;
use std::ops::{Index, IndexMut};
use std::fmt::Display;

pub struct AztecCode {
    size: usize,
    image: Vec<bool>
}

impl AztecCode {
    pub fn new(size: usize) -> AztecCode {
        if size < 11 {
            panic!("Aztec Code minimum size is 11");
        }
        let image = vec![false; size*size];
        let mut code = AztecCode { size, image };
        code.build_finder_pattern();
        code
    }

    fn build_finder_pattern(&mut self) {
        let size = self.size;
        let middle = size / 2;
        for dim in (1..11).step_by(4) {
            let start = middle - dim / 2;
            let end = middle + dim / 2;
            for i in start..=end {
                self[(i, start)] = true;
                self[(start, i)] = true;
                self[(i, end)] = true;
                self[(end, i)] = true;
            }
        }
        // top left corner (3 spaces full)
        self[(middle-5, middle-5)] = true;
        self[(middle-4, middle-5)] = true;
        self[(middle-5, middle-4)] = true;
        // top right corner (2 spaces full)
        self[(middle-5, middle+5)] = true;
        self[(middle-4, middle+5)] = true;
        // bottom right corner (1 space full)
        self[(middle+4, middle+5)] = true;
    }

    pub fn invert(&mut self) {
        for px in self.image.iter_mut() {
            *px = !*px;
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Index<(usize, usize)> for AztecCode {
    type Output = bool;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.image[row * self.size + col]
    }
}

impl IndexMut<(usize, usize)> for AztecCode {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.image[row * self.size + col]
    }
}

impl Display for AztecCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let top_bot_pad = "██".repeat(self.size + 2);
        writeln!(f, "{}", top_bot_pad)?;
        for row in 0..self.size {
            write!(f, "██")?;
            for col in 0..self.size {
                let ch = if self[(row, col)] { "  " } else { "██" };
                write!(f, "{}", ch)?;
            }
            writeln!(f, "██")?;
        }
        writeln!(f, "{}", top_bot_pad)
    }
}
