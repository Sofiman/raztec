//! Aztec Reader module

use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents the state in which the AztecReader has failed
pub enum AztecReadError {
    NotFound,
}

impl AztecReadError {
    /// Returns the error message corresponding to the current error
    pub fn message(&self) -> &str {
        use AztecReadError::*;
        match self {
            NotFound => "No Aztec Code was found"
        }
    }
}

impl Display for AztecReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AztecReadError({})", self.message())
    }
}

pub struct Marker {
    pub color: u32,
    /// loc: (col, row)
    pub loc: (usize, usize),
    /// size: (width, height)
    pub size: (usize, usize)
}

impl Marker {
    pub fn red(loc: (usize, usize), size: (usize, usize)) -> Self {
        Marker { color: 0xff0000, loc, size }
    }

    pub fn orange(loc: (usize, usize), size: (usize, usize)) -> Self {
        Marker { color: 0xffb86c, loc, size }
    }

    pub fn green(loc: (usize, usize), size: (usize, usize)) -> Self {
        Marker { color: 0x00ff00, loc, size }
    }

    pub fn blue(loc: (usize, usize), size: (usize, usize)) -> Self {
        Marker { color: 0x0000ff, loc, size }
    }
}

struct AztecCenter {
    loc: (usize, usize),
    mod_size: f32
}

pub struct AztecReader {
    width: usize,
    height: usize,
    image: Vec<bool>,
    pub markers: Vec<Marker>
}

impl AztecReader {

    /// Create a new AztecReader struct from a grayscale image
    /// 
    /// # Arguments
    /// * (`w`, `h`): The width and height of the grayscale image
    /// * `mono`: The grayscale pixel array (8 bits)
    pub fn from_grayscale((w, h): (u32, u32), mono: &[u8]) -> AztecReader {
        AztecReader { 
            width: w as usize, height: h as usize,
            image: mono.iter().map(|&x| x < 104).collect(),
            markers: vec![]
        }
    }

    fn check_ratio(&self, counts: &[usize; 9], total: usize) -> bool {
        if total < 11 {
            return false;
        }
        let mod_size = ((total - 1) / 11) as i32 + 1;
        let max_v = mod_size as i32 / 2;

        let mut total = 0;
        for &count in counts {
            if (mod_size - count as i32).abs() < max_v {
                total += 1;
            }
        }
        total >= 8
    }

    fn count_states(&self, start_row: usize, start_col: usize, drow: usize, 
        dcol: usize, row_out: &mut usize, col_out: &mut usize) -> [usize; 9] {
        let mut current_state = 0;
        let mut counts = [0usize; 9];
        let mut row = start_row;
        let mut col = start_col;
        loop {
            if row >= self.height || col >= self.width {
                break;
            }

            if self.image[row * self.width + col] { // black pixel
                if current_state % 2 == 1 {
                    current_state += 1;
                }
                counts[current_state] += 1;
            } else if current_state % 2 == 1 {
                counts[current_state] += 1;
            } else if current_state == 8 {
                break;
            } else {
                current_state += 1;
                counts[current_state] += 1;
            }

            row += drow;
            col += dcol;
        }
        *row_out = row;
        *col_out = col;
        counts
    }

    fn center_from_end(&self, counts: &[usize; 9], col: usize) -> usize {
        (col-counts[5..].iter().sum::<usize>())-counts[4] / 2
    }

    fn check_vertical(&self, start_row: usize, col: usize, mid: usize, total: usize) -> Option<usize> {
        let (mut new_row, mut new_col) = (0, 0);
        let counts = self.count_states(start_row, col, 1, 0, &mut new_row, &mut new_col);
        let total = total as i32;
        let new_total = counts.iter().sum::<usize>();
        if 9 * (total - new_total as i32).abs() < total &&
            self.check_ratio(&counts, new_total) {
            Some(self.center_from_end(&counts, new_row))
        } else {
            None
        }
    }

    fn check_horizontal(&self, row: usize, col: usize, mid: usize, total: usize) -> Option<usize> {
        todo!()
    }

    fn check_diag(&self, row: usize, col: usize, mid: usize, total: usize) -> bool {
        todo!()
    }

    fn handle_center(&mut self, counts: &[usize; 9], row: usize, col: usize,
        total: usize, centers: &mut Vec<AztecCenter>) -> Option<()> {

        let mid_val = counts[4];
        println!("verifying center...");
        let center_col = self.center_from_end(counts, col);
        self.markers.push(Marker::orange((center_col, row), (1, 1)));
        let r = row - counts[..4].iter().sum::<usize>();
        let center_row = self.check_vertical(r, center_col, mid_val, total)?;
        println!("vert OK");
        self.markers.push(Marker::red((center_col, center_row), (1, 1)));
        Some(())

        /*
        let center_col = self.check_horizontal(center_row,
            center_col, mid_val, total)?;

        if self.check_diag(center_row, center_col, counts[2], total) {
            let mod_size = total as f32 / 11.0;
            let center = AztecCenter { loc: (center_col, center_row), mod_size };
            // TODO: Check if there is any close enough centers
            centers.push(center);
            Some(())
        } else {
            None
        }*/
    }

    fn find_bullseye(&mut self) -> Result<AztecCenter, AztecReadError> {
        let mut centers = vec![];
        let mut counts = [0; 9];
        for row in 0..self.height {
            counts.fill(0);
            let mut current_state = 0;

            for col in 0..self.width {
                if self.image[row * self.width + col] { // black pixel
                    if current_state % 2 == 1 {
                        current_state += 1;
                    }
                    counts[current_state] += 1;
                } else if current_state % 2 == 1 {
                    counts[current_state] += 1;
                } else if current_state == 8 {
                    let total: usize = counts.iter().sum();
                    if self.check_ratio(&counts, total) {
                        self.handle_center(&counts, row, col, total, &mut centers);
                        //println!("check_ratio returned true");
                        counts.fill(0);
                        current_state = 0;
                    } else {
                        current_state = 7;
                        counts.rotate_left(2);
                        counts[7] = 1;
                        counts[8] = 0;
                    }
                } else {
                    current_state += 1;
                    counts[current_state] += 1;
                }
            }
        }
        if centers.is_empty() {
            Err(AztecReadError::NotFound)
        } else {
            Ok(centers.remove(0))
        }
    }

    pub fn read(&mut self) -> Result<String, AztecReadError> {
        let center = self.find_bullseye()?;
        let (center_x, center_y) = center.loc;
        println!("center: (x: {}, y: {})", center_x, center_y);
        todo!();
    }
}

