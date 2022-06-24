//! Aztec Reader module

use std::fmt::Display;
use super::{reed_solomon::ReedSolomon, AztecCode};

#[derive(Debug, Clone)]
/// Represents the state in which the AztecReader has failed
pub enum AztecReadError {
    /// The Aztec code located at the marker's position couldn't be read because
    /// it was rotated or skewed
    BadSymbolOrientation(Marker, String),

    /// The Aztec Code located at the marker's position couldn't be read because
    /// its format information was corrupted.
    CorruptedFormat(Marker, String),

    /// The Aztec code located at the marker's position couldn't be read because
    /// its message was corrupted.
    CorruptedMessage(Marker, String),
}

impl AztecReadError {
    /// Returns the error message corresponding to the current error
    pub fn message(&self) -> String {
        use AztecReadError::*;
        match self {
            BadSymbolOrientation(mk, msg) =>
                format!("Orientation detection for symbol at {} failed: {}",
                    mk, msg),
            CorruptedFormat(mk, msg) =>
                format!("Symbol at {} has an corrupted format: {}", mk, msg),
            CorruptedMessage(mk, msg) =>
                format!("Symbol at {} has an corrupted message: {}", mk, msg),
        }
    }
}

impl Display for AztecReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

#[derive(Debug, Clone)]
pub struct Marker {
    color: u32,
    /// loc: (col, row)
    loc: (usize, usize),
    /// size: (width, height)
    size: (usize, usize)
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

    /// Returns the color of the Marker area representy by the hexadecimal
    /// representation 0xRRGGBB, where R, G, B are the different color channels.
    pub fn color(&self) -> u32 { self.color }

    /// Returns the location of the top left corner of the Marker area. The
    /// location is defined by (col, row) with (col, row) respectivly
    /// corresponding to the (X, Y) axes of the input image (top left is (0,0)).
    pub fn loc(&self) -> (usize, usize) { self.loc }

    /// Returns the size of the Marker area. The size is defined by
    /// (width, height) in pixels.
    pub fn size(&self) -> (usize, usize) { self.size }
}

impl Display for Marker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, y) = self.loc;
        let (w, h) = self.size;
        write!(f, "<x: {}, y: {}, {}x{}>", x, y, w, h)
    }
}

/// An AztecCenter represents the center of an Aztec Code candidate.
pub struct AztecCenter {
    /// loc: (col, row)
    loc: (usize, usize),
    /// mod_size is in pixels
    mod_size: f32,
    corners: [(usize, usize); 4]
}

impl AztecCenter {
    /// Returns the location of the center of a possible Aztec Code as
    /// (col, row) starting at the top left corner
    pub fn loc(&self) -> (usize, usize) { self.loc }

    /// Estimated block size of the Aztec Code. Note that "block size" refer to
    /// the size in pixels of one data unit of the code.
    pub fn block_size(&self) -> f32 { self.mod_size }

    /// Estimated block size of the Aztec Code. Note that "block size" refer to
    /// the size in pixels of one data unit of the code.
    pub fn corners(&self) -> &[(usize, usize); 4] { &self.corners }

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

    /// Converts the AztecCenter in a visual marker filling the bullseye.
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
    /// Location of the top left corner
    loc: (usize, usize),
    /// Estimated Aztec code size
    size: usize,
    center: AztecCenter,
    layers: usize,
    codewords: usize,
    code_type: AztecCodeType
}

impl ReadAztecCode {

    /// Returns the location of the center of the Aztec Code. The
    /// location is defined by (col, row) with (col, row) respectivly
    /// corresponding to the (X, Y) axes of the input image (top left is (0,0)).
    pub fn location(&self) -> (usize, usize) {
        self.loc
    }

    /// The size of the Aztec Code in units.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the underlaying Aztec Code candidate (AztecCenter) of the read
    /// Aztec Code.
    pub fn center(&self) -> &AztecCenter {
        &self.center
    }

    /// Returns the type of the read Aztec Code
    pub fn code_type(&self) -> AztecCodeType {
        self.code_type
    }
}

impl Into<u8> for ReadAztecCode {
    fn into(self) -> u8 {
        assert_eq!(self.code_type, AztecCodeType::Rune,
            "Only an Aztec Rune can be turned into a byte value (cur: {:?})",
            self.code_type);
        self.codewords as u8
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
    markers: Vec<Marker>
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

    /// Returns the marker generated by the decode procedure
    pub fn markers(&self) -> &[Marker] { &self.markers }

    /// Clear the generated markers cached into the Reader
    pub fn clear_markers(&mut self) {
        self.markers.clear();
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

    fn check_vertical(&mut self, start_row: usize, col: usize, mid: usize,
        total: usize) -> Option<(usize, usize, usize)> {
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

    fn check_horizontal(&self, row: usize, start_col: usize, mid: usize,
        total: usize) -> Option<(usize, usize, usize)> {
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

    fn check_diag(&mut self, row: usize, col: usize, mid: usize, total: usize)
        -> bool {
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

    fn closest_edge_from_point(&self, mut row: usize, mut col: usize,
        (dx, dy): (isize, isize)) -> (usize, usize) {
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

            let corners = [
                // top left
                self.closest_edge_from_point(v_start, h_start, ( 0,  1)),
                // top right
                self.closest_edge_from_point(v_start, h_end,   (-1,  0)),
                // bottom_left
                self.closest_edge_from_point(v_end, h_start,   ( 1,  0)),
                // bottom right
                self.closest_edge_from_point(v_end, h_end,     ( 0, -1)),
            ];

            let ct = AztecCenter { loc: (center_col, center_row), mod_size,
                corners };
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

    /// Runs the Aztec Code finding algorithms and returns a vector of possible
    /// Aztec Code candidates (symbols that have a valid bullseye pattern). You
    /// should call the `decode` function on them. Note that this function is
    /// quite slow depending on the resolution of the input image.
    pub fn detect_codes(&mut self) -> Vec<AztecCenter> {
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

    fn sample_block(&mut self, row: usize, col: usize, w: usize, h: usize)
        -> bool {
        let mut bal: isize = 0;
        let (w2, h2) = (w / 2, h / 2);
        for i in 0..=w {
            for j in 0..=h {
                let nr = row + j;
                let nc = col + i;
                if nr >= h2         && nc >= w2 &&
                   nr < self.height && nc < self.width {
                    bal += if self.get_px(nr - h2, nc - w2) { 1 } else { -1 };
                }
            }
        }
        bal > 0
    }

    fn sample_ring(&mut self, center: &AztecCenter, radius: f32, block: usize)
        -> Vec<bool> {
        let (col, row) = center.loc;

        let dst = (center.mod_size * radius).ceil() as usize;
        let middle = dst / 2;
        if col < middle || row < middle ||
            col + middle > self.width || row + middle > self.height {
            return vec![];
        }

        let sc = dst / block + 1; // sample_count
        let bl = block / 2;
        let mut sample = vec![false; sc * 4];
        for i in 0..sc {
            let d = i * block;

            let (r, c) = (row - middle - bl, col - middle + d + block - bl);
            sample[i] = self.sample_block(r, c, bl, bl);
            self.markers.push(Marker::green((c, r), (bl, bl)));

            let (r, c) = (row - middle + d + block - bl, col + middle);
            sample[i + sc] = self.sample_block(r, c, bl, bl);
            self.markers.push(Marker::orange((c, r), (bl, bl)));

            let (r, c) = (row + middle, col - middle + d + bl - block);
            sample[sc - 1 - i + 2 * sc] = self.sample_block(r, c, bl, bl);
            self.markers.push(Marker::blue((c, r), (bl, bl)));

            let (r, c) = (row - middle + d + bl - block, col - middle - bl);
            sample[sc - 1 - i + 3 * sc] = self.sample_block(r, c, bl, bl);
            self.markers.push(Marker::red((c, r), (bl, bl)));
        }

        sample
    }

    fn get_aztec_metadata(&self, mk: Marker, code_type: &mut AztecCodeType,
        message: &[bool]) -> Result<(usize, usize), AztecReadError> {

        let (len, block_size, count) =
            if *code_type == AztecCodeType::FullSize {
                (56, 5u8, 10)
            } else {
                (40, 7u8, 7)
            };

        if message.len() != len {
            return Err(AztecReadError::CorruptedFormat(mk,
                    "Invalid message size".to_owned()));
        }
        let mut codeword = 0;
        let mut codeword_size = 0u8;
        let mut block = 0;
        let mut corner = true;
        let mut msg = vec![0; count];
        let mut i = 1;
        let mut j = 0;
        while i < len {
            if block == block_size {
                i += if corner { 3 } else { 1 };
                block = 0;
                if *code_type == AztecCodeType::FullSize {
                    corner = !corner;
                }
            } else {
                if codeword_size == 4 {
                    msg[j] = codeword;
                    j += 1;
                    codeword = 0;
                    codeword_size = 0;
                }
                block += 1;
                codeword_size += 1;
                codeword = (codeword << 1) | message[i] as usize;
                i += 1;
            }
        }
        msg[j] = codeword;

        let rs = ReedSolomon::new(4, 0b10011);
        if *code_type == AztecCodeType::FullSize {
            // indexes: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55
            // skips:   v _ _ _ _ _ v _ _  _  _  _  v  v  v  _  _  _  _  _  v  _  _  _  _  _  v  v  v  _  _  _  _  _  v  _  _  _  _  _  v  v  v  _  _  _  _  _  v  _  _  _  _  _  v v

            rs.fix_errors(&mut msg, 6)
                .map_err(|msg| AztecReadError::CorruptedFormat(mk, msg))?;

            let layers = (msg[0] << 1) | (msg[1] & 0b1000) >> 3;
            let codewords = ((msg[1] & 0b111) << 8) | (msg[2] << 4) | msg[3];
            Ok((layers + 1, codewords + 1))
        } else {
            // indexes: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
            // skips:   v _ _ _ _ _ _ _ v  v  v  _  _  _  _  _  _  _  v  v  v  _  _  _  _  _  _  _  v  v  v  _  _  _  _  _  _  _  v v

            if let Ok(_) = rs.fix_errors(&mut msg, 5) {
                let layers = msg[0] >> 2;
                let codewords = ((msg[0] & 0b11) << 4) | msg[1];
                Ok((layers + 1, codewords + 1))
            } else { // maybe it is an Aztec Rune
                for b in msg.iter_mut() {
                    *b ^= 0b1010;
                }
                rs.fix_errors(&mut msg, 5)
                    .map_err(|msg| AztecReadError::CorruptedFormat(mk, msg))?;

                *code_type = AztecCodeType::Rune;
                Ok((0, (msg[0] << 4) | msg[1]))
            }
        }
    }
    
    fn unpack_layer(&self, layer: Vec<bool>, data: &mut [usize], written: &mut usize) {

    }

    /// Tries to decode an Aztec Code candidate (AztecCenter) and retrive all
    /// its data. This may fail because of various resasons, please check the
    /// AztecReadError enum.
    pub fn decode(&mut self, center: AztecCenter)
        -> Result<ReadAztecCode, AztecReadError> {
        let (col, row) = center.loc;

        let mut code_type =
            if self.check_ring(row, col, center.mod_size, 12.3) {
                AztecCodeType::FullSize
            } else {
                AztecCodeType::Compact
            };

        let sz = center.mod_size.round() as usize;
        let r = if code_type == AztecCodeType::FullSize { 13.5 } else { 9.5 };
        let overhead_message = self.sample_ring(&center, r, sz);
        let (layers, codewords) =
            self.get_aztec_metadata(center.as_marker(), &mut code_type,
            &overhead_message)?;

        use AztecCodeType::*;
        match code_type {
            Rune => Ok(ReadAztecCode { loc: center.loc, size: 11,
                code_type, center, layers, codewords }),
            Compact => {
                println!("{} layers, {} codewords", layers, codewords);
                let size = 11 + 4 * layers;
                let mut data = vec![0usize; codewords];
                let mut written = 0;

                for l in 1..=(layers*2) { // TODO: Read content
                    let layer =
                        self.sample_ring(&center, r + 2.0 * l as f32, sz);
                    self.unpack_layer(layer, &mut data, &mut written);
                }

                Ok(ReadAztecCode { loc: center.loc, size,
                    code_type, center, layers, codewords })
            },
            FullSize => {
                println!("{} layers, {} codewords", layers, codewords);
                let raw_size = 15 + 4 * layers;
                let anchor_grid = ((raw_size - 1) / 2 - 1) / 15;
                let size = raw_size + 2 * anchor_grid;

                for l in 1..=((layers + anchor_grid) * 2) { // TODO: Read content
                    self.sample_ring(&center, r + 2.0 * l as f32, sz);
                }

                Ok(ReadAztecCode { loc: center.loc, size,
                    code_type, center, layers, codewords })
            }
        }
    }

    fn get_px(&self, row: usize, col: usize) -> bool {
        self.image[row * self.width + col]
    }

    /// Runs the Aztec Code finding algorithms and tries to decode all of them.
    /// You may not call this function every time as it is quite slow and
    /// depends a lot on the resolution of the image.
    pub fn read_codes(&mut self) -> Vec<Result<ReadAztecCode, AztecReadError>> {
        self.detect_codes().into_iter()
            .map(|center| self.decode(center)).collect()
    }
}

