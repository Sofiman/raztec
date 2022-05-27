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

fn div_ceil(a: isize, b: isize) -> isize {
    let d = a / b;
    let r = a % b;
    if (r > 0 && b > 0) || (r < 0 && b < 0) {
        d + 1
    } else {
        d
    }
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

    fn check_ratio(&self, counts: &[usize; 5], total: usize) -> bool {
        if total < 5 {
            return false;
        }
        let mod_size = div_ceil(total as isize, 5);
        let max_v = mod_size / 2;

        let mut i = 0;
        while i < 5 && (mod_size - counts[i] as isize).abs() < max_v {
            i += 1;
        }
        i == 5
    }

    fn center_from_end(&self, counts: &[usize; 5], col: usize) -> usize {
        col - counts[3..].iter().sum::<usize>() - counts[2] / 2
    }

    fn check_vertical(&mut self, start_row: usize, col: usize, mid: usize, total: usize) -> Option<usize> {
        let mut row = start_row + 1;
        let mut counts = [0usize; 5];

        // Going up to the border of the current square
        while row > 0 && self.get_px(row - 1, col) {
            counts[2] += 1;
            row -= 1;
        }
        if row == 0 {
            return None
        }

        for i in (0..2).rev() {
            let expected_col = i % 2 == 0; // when odd we count white pixels
            while row > 0 && self.get_px(row - 1, col) == expected_col
            && counts[i] <= mid {
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
            counts[2] += 1;
            row += 1;
        }
        if row > l {
            return None
        }

        for i in 3..5 {
            let expected_col = i % 2 == 0; // when odd we count white pixels
            while row <= l && self.get_px(row - 1, col) == expected_col
            && counts[i] <= mid {
                counts[i] += 1;
                row += 1;
            }
            if row > l {
                return None
            }
        }

        let total = total as i32;
        let new_total: usize = counts.iter().sum();
        if self.check_ratio(&counts, new_total) {
            Some(self.center_from_end(&counts, row - 1))
        } else {
            None
        }
    }

    fn check_horizontal(&self, row: usize, start_col: usize, mid: usize, total: usize) -> Option<usize> {
        let mut col = start_col + 1;
        let mut counts = [0usize; 5];

        // Going up to the border of the current square
        while col > 0 && self.get_px(row, col - 1) {
            counts[2] += 1;
            col -= 1;
        }
        if col == 0 {
            return None
        }

        for i in (0..2).rev() {
            let expected_col = i % 2 == 0; // when odd we count white pixels
            while col > 0 && self.get_px(row, col - 1) == expected_col
            && counts[i] <= mid {
                counts[i] += 1;
                col -= 1;
            }
            if col == 0 {
                return None
            }
        }

        let l = self.width;
        col = start_col + 2;
        while col <= l && self.get_px(row, col - 1) {
            counts[2] += 1;
            col += 1;
        }
        if col > l {
            return None
        }

        for i in 3..5 {
            let expected_col = i % 2 == 0; // when odd we count white pixels
            while col <= l && self.get_px(row, col - 1) == expected_col
            && counts[i] <= mid {
                counts[i] += 1;
                col += 1;
            }
            if col > l {
                return None
            }
        }

        let total = total as i32;
        let new_total: usize = counts.iter().sum();
        if 5 * (new_total as i32 - total).abs() < 2 * total &&
            self.check_ratio(&counts, new_total) {
            Some(self.center_from_end(&counts, col - 1))
        } else {
            None
        }
    }

    fn check_diag(&self, row: usize, col: usize, mid: usize, total: usize) -> bool {
        todo!()
    }

    fn handle_center(&mut self, counts: &[usize; 5], row: usize, col: usize,
        total: usize, centers: &mut Vec<AztecCenter>) -> Option<()> {

        let mid_val = counts[2] + counts[3] / 2;
        println!("verifying center...");
        let center_col = self.center_from_end(counts, col);
        let center_row = self.check_vertical(row, center_col, mid_val, total)?;
        println!("Vertical: OK");

        let center_col = self.check_horizontal(center_row,
            center_col, mid_val, total)?;
        self.markers.push(Marker::red((center_col, center_row), (1, 1)));
        println!("Horizontal: OK");
        Some(())
        /*
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

    fn find_bullseyes(&mut self) -> Vec<AztecCenter> {
        let mut centers = vec![];
        let mut counts = [0; 5];
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
                } else if current_state == 4 {
                    let total: usize = counts.iter().sum();
                    if self.check_ratio(&counts, total) &&
                        self.handle_center(&counts, row, col, total,
                            &mut centers).is_some() {
                            counts.fill(0);
                            current_state = 0;
                    } else {
                        current_state = 3;
                        counts.rotate_left(2);
                        counts[3] = 1;
                        counts[4] = 0;
                    }
                } else {
                    current_state += 1;
                    counts[current_state] += 1;
                }
            }
        }
        centers
    }

    fn get_px(&self, row: usize, col: usize) -> bool {
        self.image[row * self.width + col]
    }

    pub fn read(&mut self) -> Result<String, AztecReadError> {
        let centers = self.find_bullseyes();
        if centers.is_empty() {
            return Err(AztecReadError::NotFound);
        }
        for center in centers {
            let (center_x, center_y) = center.loc;
            println!("center: (x: {}, y: {})", center_x, center_y);
        }
        todo!();
    }
}

