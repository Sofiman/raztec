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
    compact: bool,
    image: Vec<bool>
}

impl AztecCode {
    pub fn new(compact: bool, size: usize) -> AztecCode {
        if compact {
            if size < 11 {
                panic!("Compact Aztec Code has a minimum size of 11x11");
            }
        } else if size < 15 {
            panic!("Full-size Aztec Code has a minimum size of 15x15");
        }

        let image = vec![false; size*size];
        let mut code = AztecCode { size, compact, image };
        code.build_finder_pattern();
        code
    }

    fn build_finder_pattern(&mut self) {
        let size = self.size;
        let middle = size / 2;
        let max_ring_size = if self.compact { 11 } else { 15 };

        for dim in (1..max_ring_size).step_by(4) {
            let start = middle - dim / 2;
            let end = middle + dim / 2;
            for i in start..=end {
                self[(i, start)] = true;
                self[(start, i)] = true;
                self[(i, end)] = true;
                self[(end, i)] = true;
            }
        }

        let d2u = max_ring_size / 2; // max ring size divided per 2
        let d2l = d2u - 1; // max ring size divided per 2 minus 1

        // top left corner (3 spaces full)
        self[(middle-d2u, middle-d2u)] = true;
        self[(middle-d2l, middle-d2u)] = true;
        self[(middle-d2u, middle-d2l)] = true;
        // top right corner (2 spaces full)
        self[(middle-d2u, middle+d2u)] = true;
        self[(middle-d2l, middle+d2u)] = true;
        // bottom right corner (1 space full)
        self[(middle+d2l, middle+d2u)] = true;

        // generate anchor grid
        if !self.compact {
            // precompute modulo to avoid convertion to i32
            let mid = middle % 16;
            for x in 0..size {
                if x % 16 == mid {
                    for y in 0..size {
                        self[(x, y)] = (x + y + 1) % 2 == 1;
                    }
                } else {
                    for y in 0..size {
                        if y % 16 == mid {
                            self[(x, y)] = (x + y + 1) % 2 == 1;
                        }
                    }
                }
            }
        }
    }

    pub fn invert(&mut self) {
        for px in self.image.iter_mut() {
            *px = !*px;
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_compact(&self) -> bool {
        self.compact
    }

    pub fn to_rgb(&self, module_size: usize) -> Vec<u8> {
        let side = self.size * module_size;
        let mut pixels = vec![0; side * side];
        for i in 0..side {
            for j in 0..side {
                let bit = self.image[(i / module_size) * self.size + j / module_size];
                pixels[i * side + j] = if bit { 0 } else { 255 };
            }
        }

        pixels
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
