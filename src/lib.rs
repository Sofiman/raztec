//! # raztec
//!
//! A library for reading and writing Aztec 2D Barcodes using no external lib

pub mod writer;
#[allow(warnings)]
pub mod reader;
pub mod reed_solomon;
use std::ops::{Index, IndexMut};
use std::fmt::Display;

/// Represents a raw Aztec Code.
///
/// You can preview the AztecCode in the console using the standard formatter
/// as this struct implements the trait Display.
#[derive(Clone, PartialEq, Eq)]
pub struct AztecCode {
    size: usize,
    compact: bool,
    image: Vec<bool>
}

impl AztecCode {

    /// Create a new empty Aztec Code (with the bullseye).
    ///
    /// # Arguments
    ///
    /// * `compact` - Indicates if this Aztec Code is compact or full-size
    /// * `size` - Aztec Code's side size, compact Aztec Codes have a minimum
    /// size of 11x11 and full-size Aztec Codes of 15x15
    pub fn new(compact: bool, size: usize) -> AztecCode {
        if compact {
            if size < 11 {
                panic!("Compact Aztec Codes have a minimum size of 11x11");
            }
        } else if size < 15 {
            panic!("Full-size Aztec Codes have a minimum size of 15x15");
        }

        let image = vec![false; size * size];
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

    /// Invert all Aztec Code cells' state.
    pub fn invert(&mut self) {
        for px in self.image.iter_mut() {
            *px = !*px;
        }
    }

    /// Returns the size of the side of this Aztec Code.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns true if this is a compact Aztec Code, false otherwise.
    pub fn is_compact(&self) -> bool {
        self.compact
    }

    /// Convert the AztecCode to an image
    ///
    /// # Arguments
    ///
    /// * `module_size` - Scaling factor of each Aztec Code cell
    /// * `empty_color` - Color that represents a `false` (Normally white)
    /// * `filled_color` - Color that represents a `true` (Normally black)
    pub fn to_image<C: Clone>(&self, module_size: usize,
        empty_color: C, filled_color: C) -> Vec<C> {
        let size = self.size;
        let side = size * module_size;
        let mut pixels = vec![empty_color.clone(); side * side];
        for i in 0..side {
            for j in 0..side {
                let b = self.image[(i / module_size) * size + j / module_size];
                pixels[i * side + j] = if b {
                    filled_color.clone() 
                } else { 
                    empty_color.clone()
                };
            }
        }

        pixels
    }

    /// Creates a new RGB8 image from the raw Aztec Code data. Black represent
    /// trues and White represent falses.
    pub fn to_rgb8(&self, module_size: usize) -> Vec<u32> {
        self.to_image(module_size, 0xffffff, 0)
    }


    /// Creates a new MONO8 (black and white) image from the raw Aztec Code
    /// data. Black represent trues and White represent falses.
    pub fn to_mono8(&self, module_size: usize) -> Vec<u8> {
        self.to_image(module_size, 255, 0)
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
