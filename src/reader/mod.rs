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
        total == 9
    }

    fn center_from_end(&self, counts: &[usize; 9], col: usize) -> usize {
        col - counts[5..].iter().sum::<usize>() - counts[4] / 2
    }

    fn check_vertical(&mut self, start_row: usize, col: usize, mid: usize, total: usize) -> Option<usize> {
        let mut row = start_row + 1;
        let mut counts = [0usize; 9];

        // Going up to the border of the current square
        while row > 0 && self.get_px(row - 1, col) {
            self.markers.push(Marker::blue((col, row - 1), (1, 1)));
            counts[4] += 1;
            row -= 1;
        }
        if row == 0 {
            return None
        }

        for i in (0..4).rev() {
            let expected_col = i % 2 == 0; // when odd we count white pixels
            while row > 0 && self.get_px(row - 1, col) == expected_col
            && counts[i] <= mid {
                self.markers.push(Marker::green((col, row - 1), (1, 1)));
                counts[i] += 1;
                row -= 1;
            }
            if row == 0 {
                return None
            }
        }

        let l = self.height;
        row = start_row + 2;
        while row <= l && self.get_px(row - 1, col) {
            self.markers.push(Marker::blue((col, row - 1), (1, 1)));
            counts[4] += 1;
            row += 1;
        }
        if row > l {
            return None
        }

        for i in 5..9 {
            let expected_col = i % 2 == 0; // when odd we count white pixels
            while row <= l && self.get_px(row - 1, col) == expected_col
            && counts[i] <= mid {
                self.markers.push(Marker::green((col, row - 1), (1, 1)));
                counts[i] += 1;
                row += 1;
            }
            if row > l {
                return None
            }
        }

        let total = total as i32;
        let new_total: usize = counts.iter().sum();
        if 9 * (new_total as i32 - total).abs() < 2 * total &&
            self.check_ratio(&counts, new_total) {
            Some(self.center_from_end(&counts, row - 1))
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

        let mid_val = (counts.iter().sum::<usize>() as f32 / 9.0).ceil() as usize;
        println!("verifying center...");
        let center_col = self.center_from_end(counts, col);
        self.markers.push(Marker::orange((center_col, row), (1, 1)));
        let center_row = self.check_vertical(row, center_col, mid_val, total)?;
        println!("Vertical: OK");
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
                if self.get_px(row, col) { // black pixel
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

    fn get_px(&self, row: usize, col: usize) -> bool {
        self.image[row * self.width + col]
    }

    pub fn read(&mut self) -> Result<String, AztecReadError> {
        let center = self.find_bullseye()?;
        let (center_x, center_y) = center.loc;
        println!("center: (x: {}, y: {})", center_x, center_y);
        todo!();
    }
}

