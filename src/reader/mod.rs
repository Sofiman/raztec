//! Aztec Reader module

use std::fmt::Display;
use super::{reed_solomon::ReedSolomon, AztecCode};

#[derive(Debug, Clone)]
/// Represents the state in which the AztecReader has failed
pub enum AztecReadError {
    BadSymbolOrientation(Marker, String),
    InvalidSize(Marker, String)
}

impl AztecReadError {
    /// Returns the error message corresponding to the current error
    pub fn message(&self) -> String {
        use AztecReadError::*;
        match self {
            BadSymbolOrientation(mk, msg) =>
                format!("Orientation detection for symbol at {} failed: {}",
                    mk, msg),
            InvalidSize(mk, msg) =>
                format!("Symbol at {} has an invalid size: {}", mk, msg)
        }
    }
}

impl Display for AztecReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AztecReadError({})", self.message())
    }
}

#[derive(Debug, Clone)]
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

    pub fn pink(loc: (usize, usize), size: (usize, usize)) -> Self {
        Marker { color: 0xff79c6, loc, size }
    }
}

impl Display for Marker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, y) = self.loc;
        let (w, h) = self.size;
        write!(f, "<x: {}, y: {}, {}x{}>", x, y, w, h)
    }
}

pub struct AztecCenter {
    /// loc: (col, row)
    loc: (usize, usize),
    /// mod_size is in pixels
    mod_size: f32
}

impl AztecCenter {
    pub fn location(&self) -> (usize, usize) {
        self.loc
    }

    fn dst_sqd(&self, other: &AztecCenter) -> usize {
        let (ax, ay) = self.loc;
        let (bx, by) = other.loc;
        let (dx, dy) = (bx as isize - ax as isize, by as isize - ay as isize);
        (dx * dx) as usize + (dy * dy) as usize
    }

    fn avg_with(&mut self, other: &AztecCenter) {
        let (ax, ay) = self.loc;
        let (bx, by) = other.loc;
        self.loc = ((ax + bx) / 2, (ay + by) / 2);
        self.mod_size = (self.mod_size + other.mod_size) / 2.0;
    }

    pub fn as_marker(&self) -> Marker {
        let (x, y) = self.loc;
        let size = (self.mod_size * 3.5).ceil() as usize;
        Marker::orange((x - size / 2, y - size / 2), (size, size))
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AztecCodeType {
    Rune,
    Compact,
    FullSize
}

pub struct ReadAztecCode {
    loc: (usize, usize),
    size: (usize, usize),
    center: AztecCenter,
    layers: usize,
    codewords: usize,
    code_type: AztecCodeType
}

impl ReadAztecCode {

    pub fn location(&self) -> (usize, usize) {
        self.loc
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn center(&self) -> &AztecCenter {
        &self.center
    }

    pub fn code_type(&self) -> AztecCodeType {
        self.code_type
    }
}

impl Into<String> for ReadAztecCode {
    fn into(self) -> String {
        todo!("Implement into string")
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

    fn check_vertical(&mut self, start_row: usize, col: usize, mid: usize, total: usize) -> Option<(usize, usize, usize)> {
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
        let start = row;

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
        if (new_total as i32 - total).abs() < 2 * total &&
            self.check_ratio(&counts, new_total) {
            Some((start, self.center_from_end(&counts, row - 1), row - 2))
        } else {
            None
        }
    }

    fn check_horizontal(&self, row: usize, start_col: usize, mid: usize, total: usize) -> Option<(usize, usize, usize)> {
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
        let start = col;

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
        if (new_total as i32 - total).abs() < 2 * total &&
            self.check_ratio(&counts, new_total) {
            Some((start, self.center_from_end(&counts, col - 1), col - 2))
        } else {
            None
        }
    }

    fn check_diag(&mut self, row: usize, col: usize, mid: usize, total: usize) -> bool {
        let mut counts = [0usize; 5];
        let mut i = 0;

        // Going up to the border of the current square
        while row >= i && col >= i && self.get_px(row - i, col - i) {
            counts[2] += 1;
            i += 1;
        }
        if row < i || col < i {
            return false;
        }

        for k in (0..2).rev() {
            let expected_col = k % 2 == 0; // when odd we count white pixels
            while row >= i && col >= i && self.get_px(row - i, col - i)
                == expected_col && counts[k] <= mid {
                counts[k] += 1;
                i += 1;
            }
            if row < i || col < i || counts[k] > mid {
                return false;
            }
        }

        let (l, h) = (self.width, self.height);
        i = 1;
        while row+i < h && col+i < l && self.get_px(row + i, col + i) {
            counts[2] += 1;
            i += 1;
        }
        if row + i >= h || col + i >= l {
            return false;
        }

        for k in 3..5 {
            let expected_col = k % 2 == 0; // when odd we count white pixels
            while row+i < h && col+i < l && self.get_px(row + i, col + i)
                == expected_col && counts[k] <= mid {
                counts[k] += 1;
                i += 1;
            }
            if row + i >= h || col + i >= l || counts[k] > mid {
                return false;
            }
        }

        let total = total as i32;
        let new_total: usize = counts.iter().sum();
        (new_total as i32 - total).abs() < 2 * total &&
            self.check_ratio(&counts, new_total)
    }

    fn check_ring(&mut self, row: usize, col: usize, mod_size: f32,
        ring_size: f32) -> bool {
        let dst = (mod_size * ring_size).ceil() as usize;
        let middle = dst / 2;
        if col < middle || row < middle ||
            col + middle > self.width || row + middle > self.height {
            return false;
        }

        let mut counts = [0; 4];
        for d in 0..dst {
            self.markers.push(Marker::green((col - middle + d, row - middle), (1,1)));
            self.markers.push(Marker::red((col - middle, row - middle + d), (1,1)));
            self.markers.push(Marker::orange((col + middle, row - middle + d), (1,1)));
            self.markers.push(Marker::blue((col - middle + d, row + middle), (1,1)));
            counts[0] += self.get_px(row - middle, col - middle + d) as usize;
            counts[1] += self.get_px(row - middle + d, col - middle) as usize;
            counts[2] += self.get_px(row - middle + d, col + middle) as usize;
            counts[3] += self.get_px(row + middle, col - middle + d) as usize;
        }

        let target = (dst * 3) / 4;
        let mut i = 0;
        while i < 4 && counts[i] >= target {
            i += 1;
        }
        i == 4
    }

    fn closest_edge_from_point(&self, mut row: usize, mut col: usize, (dx, dy): (isize, isize)) -> (usize, usize) {
        if row == 0 || col == 0 || row >= self.height || col >= self.width {
            return (row, col);
        }
        while row > 0 && col > 0 && row < self.height && col < self.width &&
            !self.get_px(row, col) {
            row = (row as isize + dy) as usize;
            col = (col as isize + dx) as usize;
        }
        (row, col)
    }

    fn handle_center(&mut self, counts: &[usize; 5], row: usize, col: usize,
        total: usize, centers: &mut Vec<AztecCenter>) -> Option<()> {

        let mid_val = counts[2] + counts[3] / 2;
        let center_col = self.center_from_end(counts, col);
        let (v_start, center_row, v_end) =
            self.check_vertical(row, center_col, mid_val, total)?;
        let (h_start, center_col, h_end) =
            self.check_horizontal(center_row, center_col, mid_val, total)?;

        let mod_size = total as f32 / 5.0;
        if self.check_diag(center_row, center_col, mid_val, total)
            && self.check_ring(center_row, center_col, mod_size, 4.0){

            /*
            self.markers.push(Marker::green((h_start, v_start), (1,1)));
            let (x, y) = self.closest_edge_from_point(v_start, h_start, (0, 1));
            println!("top_left: {:?} => {:?}", (v_start, h_start), (x, y));
            self.markers.push(Marker::green((y, x), (1,1)));

            self.markers.push(Marker::red((h_end, v_start), (1,1)));
            let (x, y) = self.closest_edge_from_point(v_start, h_end, (-1, 0));
            println!("top_right: {:?} => {:?}", (v_start, h_end), (x, y));
            self.markers.push(Marker::red((y, x), (1,1)));

            self.markers.push(Marker::orange((h_start, v_end), (1,1)));
            let (x, y) = self.closest_edge_from_point(v_end, h_start, (1, 0));
            println!("bottom_left: {:?} => {:?}", (h_start, v_end), (x, y));
            self.markers.push(Marker::orange((y, x), (1,1)));

            self.markers.push(Marker::blue((h_end, v_end), (1,1)));
            let (x, y) = self.closest_edge_from_point(v_end, h_end, (0, -1));
            println!("bottom_right: {:?} => {:?}", (v_end, h_end), (x, y));
            self.markers.push(Marker::blue((y, x), (1,1)));
            */

            let ct = AztecCenter { loc: (center_col, center_row), mod_size };
            for other in centers.iter_mut() {
                if ct.dst_sqd(other) < 100 { // less than 10 pixels apart
                    other.avg_with(&ct);
                    return Some(());
                }
            }
            centers.push(ct);
            Some(())
        } else {
            None
        }
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

    fn sample_block(&mut self, row: usize, col: usize, w: usize, h: usize) -> bool {
        let mut bal: isize = 0;
        let (w2, h2) = (w / 2, h / 2);
        for i in 0..=w {
            for j in 0..=h {
                let nrow = row + j;
                let ncol = col + i;
                if nrow >= h2 && ncol >= w2 && nrow < self.height && ncol < self.width {
                    self.markers.push(Marker::pink((ncol - h2, nrow - w2), (w, h)));
                    bal += if self.get_px(nrow - h2, ncol - w2) { 1 } else { -1 };
                }
            }
        }
        bal > 0
    }

    fn sample_ring(&mut self, center: &AztecCenter, radius: f32, block: usize) -> Vec<bool> {
        let (col, row) = center.loc;

        let dst = (center.mod_size * radius).ceil() as usize;
        let middle = dst / 2;
        if col < middle || row < middle ||
            col + middle > self.width || row + middle > self.height {
            return vec![];
        }

        let sample_count = dst / block + 1;
        let bl = block / 2;
        let mut sample = vec![false; sample_count * 4];
        for i in 0..sample_count {
            let d = i * block;

            sample[i] = self.sample_block(row - middle - bl, col - middle + d - bl, bl, bl);
            self.markers.push(Marker::green((col - middle + d - bl, row - middle - bl), (bl, bl)));

            sample[i + sample_count] = self.sample_block(row - middle + d - bl, col + middle, bl, bl);
            self.markers.push(Marker::orange((col + middle, row - middle + d - bl), (bl, bl)));

            sample[i + 2 * sample_count] = self.sample_block(row + middle, col - middle + d + bl, bl, bl);
            self.markers.push(Marker::blue((col - middle + d + bl, row + middle), (bl, bl)));

            sample[i + 2 * sample_count] = self.sample_block(row - middle + d + bl, col - middle - bl, bl, bl);
            self.markers.push(Marker::red((col - middle - bl, row - middle + d + bl), (bl, bl)));
        }

        sample
    }

    fn to_binary(arr: &[bool]) -> usize {
        arr.iter().fold(0, |acc, &v| (acc << 1) | if v { 1 } else { 0 })
    }

    fn get_aztec_metadata(&self, code_type: AztecCodeType, message: &[bool]) -> Result<(usize, usize, AztecCodeType), AztecReadError> {
        if code_type == AztecCodeType::FullSize {
            let layers = Self::to_binary(&message[2..7]);
            let codewords =
                  (Self::to_binary(&message[8..13]) << 6) 
                | (Self::to_binary(&message[16..20]) << 2) 
                |  Self::to_binary(&message[21..23]);
            println!("layers: {} ({:#04b})", layers + 1, layers);
            println!("codewods: {} ({:#013b})", codewords + 1, codewords);
            Ok((layers + 1, codewords + 1, code_type))
        } else {
            let mut ecc: Vec<usize> = message[10..].chunks(4)
                .map(|x| Self::to_binary(x) as usize).collect();

            let rs = ReedSolomon::new(4, 0b10011);
            if let Ok(_) = rs.fix_errors(&mut ecc[..], 2) {
                let layers = Self::to_binary(&message[2..4]);
                let codewords = Self::to_binary(&message[4..9]) << 1 |
                    message[11] as usize;

                println!("layers: {} ({:#04b})", layers + 1, layers);
                println!("codewods: {} ({:#08b})", codewords + 1, codewords);
                Ok((layers + 1, codewords + 1, AztecCodeType::Compact))
            } else {
                Ok((0, 0, AztecCodeType::Rune))
            }
        }
    }

    fn process_entry(&mut self, center: AztecCenter) 
        -> Result<ReadAztecCode, AztecReadError> {
        let (col, row) = center.loc;

        let code_type = if self.check_ring(row, col, center.mod_size, 12.3) {
            AztecCodeType::FullSize
        } else {
            AztecCodeType::Compact
        };

        let pos = if code_type == AztecCodeType::FullSize { 13.5 } else { 9.5 };
        let overhead_message = self.sample_ring(&center, pos,
            center.mod_size.floor() as usize);
        let (layers, codewords, code_type) =
            self.get_aztec_metadata(code_type, &overhead_message)?;

        // TODO: Read content

        let sz = center.mod_size as usize;
        Ok(ReadAztecCode { loc: center.loc, size: (sz, sz),
            code_type, center, layers, codewords })
    }

    fn get_px(&self, row: usize, col: usize) -> bool {
        self.image[row * self.width + col]
    }

    pub fn read(&mut self) -> Vec<Result<ReadAztecCode, AztecReadError>> {
        self.find_bullseyes().into_iter()
            .map(|center| self.process_entry(center)).collect()
    }
}

