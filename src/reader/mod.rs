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

/// Marks a specific feature in an image
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

    /// Detected corners of the center of the bullseye
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

/// Supported Aztec Code types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AztecCodeType {
    /// An Aztec Rune is a special Compact Aztec Code that can hold a single
    /// byte of information while having a very small footprint
    Rune,

    /// A Compact Aztec Code is a type of Aztec Code constructed by 1 to 4
    /// layers (608 bits available for 4 layers, for examples: 110 numbers, 89
    /// signs (text), 53 bytes). Note that a Compact Aztec Code can hold more
    /// data than a Full-Size Aztec Code and therefore Full-Size codes with at
    /// most 4 layers are rarely used.
    Compact,

    /// A Full-Size Aztec Code can be constructed by 1 to 32 layers which can
    /// hold up to 3832 digits, 3067 letters, or 1914 bytes of data.
    FullSize
}

/// Aztec Code Special features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AztecCodeFeature {
    /// Represents an ECI escape code at the specified index
    ECI{index: usize, n: usize},

    /// Represents an FNC1 escape code in the data stream at the specified
    /// index. This escape symbol is used to mark the presence of an GS1 AI
    /// (Application Identifier). Note that this feature may appear multiple
    /// times in a single code, see GS1 barcode standards for examples.
    FNC1{index: usize}
}

pub struct ReadAztecCode {
    /// Location of the top left corner
    loc: (usize, usize),
    /// Estimated Aztec code size
    size: usize,
    center: AztecCenter,
    layers: usize,
    codewords: usize,
    code_type: AztecCodeType,
    data: Vec<u8>,
    features: Vec<AztecCodeFeature>
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

    /// Returns the number of layers of the Aetc Code
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Returns the number of codewords stored in the read Aztec Code
    pub fn codewords(&self) -> usize {
        self.codewords
    }

    /// Returns all the features found in the read Aztec Code
    pub fn features(&self) -> &[AztecCodeFeature] {
        &self.features
    }

    /// Returns the data held by the read Aztec Code
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consumes the ReadAztecCode and returns its data
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Direction {
    None,
    Down,
    Left,
    Up,
    Right,
}

impl Default for Direction {
    fn default() -> Self {
        Direction::None
    }
}

#[derive(Debug, Clone, Default)]
struct Domino {
    dir: Direction,
    head_pos: (usize, usize),
    head: bool,
    tail: bool,
    offset: usize
}

impl Domino {

    fn down(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Down,  ..Default::default() }
    }

    fn left(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Left,  ..Default::default() }
    }

    fn up(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Up,    ..Default::default() }
    }

    fn right(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Right, ..Default::default() }
    }

    fn head(&self) -> (usize, usize) {
        self.head_pos
    }

    fn tail(&self) -> (usize, usize) {
        let (row, col) = self.head_pos;
        match &self.dir {
            Direction::None  => unreachable!("tail() on an empty domino"),
            Direction::Down  => (row + 1 + self.offset, col),
            Direction::Left  => (row, col - 1 - self.offset),
            Direction::Up    => (row - 1 - self.offset, col),
            Direction::Right => (row, col + 1 + self.offset)
        }
    }

    fn check_splitting(&mut self, middle: usize) -> usize {
        let (x, y) = self.tail();
        if x % 16 == middle || y % 16 == middle {
            self.offset += 1;
        }
        self.offset
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Upper,
    Lower,
    Mixed,
    Punctuation,
    Digit
}

impl Mode {
    fn capacity(&self) -> usize {
        match self {
            Mode::Digit => 4,
            _ => 5
        }
    }
}

struct AztecReader {
    size: usize,
    layers: usize,
    compact: bool,
    dominos: Vec<Domino>,
    codewords: usize,
}

type ReaderResults = Result<(Vec<u8>, Vec<AztecCodeFeature>), String>;

impl AztecReader {
    fn new(codewords: usize, layers: usize) -> Self {
        let compact = layers <= 4;
        let bullseye_size = if compact { 11 } else { 15 };
        let raw_size = layers * 4 + bullseye_size;
        let size = if compact {
            raw_size
        } else {
            raw_size + 2 * (((raw_size - 1) / 2 - 1) / 15)
        };
        // Compute the expected number of dominos in the final Aztec Code
        let len = (raw_size * raw_size - bullseye_size * bullseye_size) / 2;
        let mut dominos = vec![Domino::default(); len];

        if compact {
            let mut idx = 0;
            for layer in 0..layers {
                let start = 2 * layer;
                let end = size - start - 1;
                let limit = end - start - 1;

                for i in 0..limit {
                    let base = idx + i;
                    dominos[base          ] = Domino::right((start + i,start));
                    dominos[base + limit  ] = Domino::up((end, start + i));
                    dominos[base + limit*2] = Domino::left((end - i, end));
                    dominos[base + limit*3] = Domino::down((start, end - i));
                }
                idx += limit * 4;
            }
        } else {
            let mid = (size / 2) % 16;
            let mut idx = 0;
            let mut skips = 0;
            let mut dpl = (layers - 5) * 4 + 32; // domino per line
            for layer in 0..layers {
                let start = 2 * layer;
                let end = size - start - 1;
                let mut limit = end - start - 1;

                if (start + skips) % 16 == mid {
                    skips += 1;
                }

                let mut delta = 0; // check if any domino will be cut in half
                if (start + skips + 1) % 16 == mid {
                    delta = 1;
                    limit -= 1;
                }

                let mut j = idx;
                for offset in 0..(limit - skips * 2) {
                    if (start + offset + skips) % 16 != mid {
                        let mut domino = Domino::right(
                            (start + offset + skips, start + skips));
                        domino.check_splitting(mid);
                        dominos[j] = domino;

                        domino = Domino::up(
                            (end - skips, start + offset + skips));
                        domino.check_splitting(mid);
                        dominos[j + dpl] = domino;

                        domino = Domino::left(
                            (end - offset - skips, end - skips));
                        domino.check_splitting(mid);
                        dominos[j + dpl * 2] = domino;

                        domino = Domino::down(
                            (start + skips, end - offset - skips));
                        domino.check_splitting(mid);
                        dominos[j + dpl * 3] = domino;
                        j += 1;
                    }
                }
                idx += dpl * 4;
                dpl -= 4;
                skips += delta;
            }
        }

        AztecReader { size, layers, dominos, codewords, compact }
    }

    /// Extract the data contained in the Aztec Code into the reader's dominos
    fn extract(&mut self, code: AztecCode) {
        assert!(code.size() == self.size,
            "Attempt to extract a bigger or smaller than expected Aztec Code");
        for domino in self.dominos.iter_mut() {
            if domino.dir == Direction::None {
                continue;
            }
            domino.head = code[domino.head()];
            domino.tail = code[domino.tail()];
        }
    }

    fn as_words(&self, skip: usize, codeword_size: usize) -> Vec<usize> {
        let dpc = codeword_size / 2; // domino_per_codeword
        let mut words = Vec::with_capacity(self.dominos.len() / dpc);
        for codeword in self.dominos[skip..].chunks(dpc) {
            words.push(codeword.iter().fold(0, |acc, dom| {
                ((dom.head as usize) << 1) | (dom.tail as usize) | (acc << 2)
            }))
        }
        words
    }

    fn remove_bitstuffing(bitstr: &mut Vec<bool>, codeword_size: usize) {
        if bitstr.len() < codeword_size {
            return;
        }
        let mut i = 0;
        let mut l = bitstr.len() - codeword_size;
        let limit = codeword_size - 1;
        while i < l {
            let first = bitstr[i];
            let mut j = 1;
            while j < limit && bitstr[i + j] == first {
                j += 1;
            }
            if j == limit {
                i += limit;
                l -= 1;
                bitstr.remove(i);
            } else {
                i += codeword_size;
            }
        }
    }

    fn get_word_value(range: &[bool]) -> usize {
        range.iter().fold(0, |acc, &val| (acc << 1) | (val as usize))
    }

    /// Handle the Word B/S (byte sequence) in the data stream
    fn handle_bytes(bitstr: &[bool], mut i: usize, data: &mut Vec<u8>) -> usize{
        if i + 5 >= bitstr.len() {
            // not enough data for the length of the byte shift sequence so
            // ignore the B/S word, it is probably end padding ones
            return i;
        }
        let mut len = Self::get_word_value(&bitstr[i..i+5]);
        i += 5;
        if len == 0 {
            if i + 11 >= bitstr.len() {
                return i; // not enough data for the 11-bit size indicator
            }
            len = Self::get_word_value(&bitstr[i..i+11]) + 31;
            i += 11;
        }
        let end = i + len * 8;
        if end > bitstr.len() {
            return i; // not enough data for `len` bytes
        }
        data.extend(
            bitstr[i..end].chunks(8).map(|r| Self::get_word_value(r) as u8)
        );
        end
    }

    /// Handle the Word Flg(n) (ECI or FNC1) in the data stream
    fn handle_flg(bitstr: &[bool], pos: usize, mut i: usize,
        features: &mut Vec<AztecCodeFeature>) -> usize {
        let n = Self::get_word_value(&bitstr[i..i+3]);
        i += 3;
        if n == 0 {
            features.push(AztecCodeFeature::FNC1{index: pos});
            return i;
        }
        let mut n = 0;
        let end = i + n * 4;
        for j in (i..end).step_by(4) {
            let digit = Self::get_word_value(&bitstr[j..j+4]);
            if !(2..11).contains(&digit) {
                return end; // invalid digit
            }
            n = n * 10 + digit - 2;
        }
        features.push(AztecCodeFeature::ECI{index: pos, n});
        end
    }

    /// Extract the data and features of the Read Aztec Code
    fn to_words(bitstr: &[bool]) -> ReaderResults {
        let l = bitstr.len();
        let mut words = vec![];
        let mut features = vec![];

        let mut mode = Mode::Upper;
        let mut shift_mode = None;

        let mut i = 0;
        let mut next = mode.capacity(); // next idx
        while next <= l {
            let word = Self::get_word_value(&bitstr[i..next]);
            let current_mode = shift_mode.take().unwrap_or(mode);
            match current_mode {
                Mode::Upper | Mode::Lower => match word {
                    0 => shift_mode = Some(Mode::Punctuation),
                    1 => words.push(b' '),
                    2..=27 => {
                        let offset = if current_mode == Mode::Upper {
                            b'A'
                        } else {
                            b'a'
                        };
                        words.push(offset + word as u8 - 2);
                    }
                    28 => {
                        if current_mode == Mode::Upper {
                            mode = Mode::Lower;
                        } else {
                            shift_mode = Some(Mode::Upper);
                        }
                    },
                    29 => mode = Mode::Mixed,
                    30 => mode = Mode::Digit,
                    31 => next = Self::handle_bytes(bitstr, next, &mut words),
                    _ => return Err(format!("Invalid word {} in {:?} mode",
                        word, current_mode))
                },
                Mode::Mixed => match word {
                    0 => shift_mode = Some(Mode::Punctuation),
                    1 => words.push(b' '),
                    2..=14 => words.push(1 + word as u8 - 2),
                    15..=19 => words.push(27 + word as u8 - 15),
                    20..=27 => words.push(
                        [b'@', b'\\', b'^', b'_', b'`', b'|', b'~', 127]
                        [word as usize - 20]
                    ),
                    28 => mode = Mode::Lower,
                    29 => mode = Mode::Upper,
                    30 => mode = Mode::Punctuation,
                    31 => next = Self::handle_bytes(bitstr, next, &mut words),
                    _ => return Err(format!("Invalid word {} in Mixed mode",
                            word))
                },
                Mode::Punctuation => match word {
                    0 => next = Self::handle_flg(bitstr, words.len(), next,
                            &mut features),
                    1 => words.push(b'\r'),
                    2 => words.extend([b'\r', b'\n']),
                    3 => words.extend([b'.', b' ']),
                    4 => words.extend([b',', b' ']),
                    5 => words.extend([b':', b' ']),
                    6..=20 => words.push(b'!' + word as u8 - 6),
                    21..=26 => words.push(b':' + word as u8 - 21),
                    27..=30 => words.push(
                        [b'[', b']', b'{', b'}']
                        [word as usize - 27]
                    ),
                    31 => mode = Mode::Upper,
                    _ => return Err(format!(
                            "Invalid word {} in Punctuation mode", word))
                },
                Mode::Digit => match word {
                    0 => shift_mode = Some(Mode::Punctuation),
                    1 => words.push(b' '),
                    2..=11 => words.push(b'0' + word as u8 - 2),
                    12 => words.push(b','),
                    13 => words.push(b'.'),
                    14 => mode = Mode::Upper,
                    15 => shift_mode = Some(Mode::Upper),
                    _ => return Err(format!("Invalid word {} in Digit mode",
                            word))
                }
            }
            i = next;
            next += shift_mode.unwrap_or(mode).capacity();
        }

        Ok((words, features))
    }

    /// Decodes the Aztec Code into the couple (extracted_data, features).
    fn read(self) -> ReaderResults {
        let (codeword_size, prim) = match self.layers {
             1..=2  => ( 6,       0b1000011),
             3..=8  => ( 8,     0b100101101),
             9..=22 => (10,   0b10000001001),
            23..=32 => (12, 0b1000001101001),
            x => return Err(format!("Aztec code with {} layers is illegal", x))
        };
        let bits_in_layers =
            (if self.compact { 88 } else { 112 } + 16 * self.layers)
            * self.layers;
        let start_align = (bits_in_layers % codeword_size) / 2;

        let mut words = self.as_words(start_align, codeword_size);
        let rs = ReedSolomon::new(codeword_size as u8, prim);

        let nb_check_words = (bits_in_layers / codeword_size) - self.codewords;
        rs.fix_errors(&mut words, nb_check_words)?;

        let mut bitstr = Vec::with_capacity(self.codewords * codeword_size);
        for byte in words[..self.codewords].iter() {
            // convert [corrected] the codewords into a bitstring
            bitstr.extend((0..codeword_size).rev()
                .map(|bit| ((byte >> bit) & 1) == 1));
        }
        Self::remove_bitstuffing(&mut bitstr, codeword_size);
        //println!("{:?}", bitstr.iter().map(|&x| if x { '1' } else { '0' }).collect::<String>());
        Self::to_words(&bitstr)
    }

}

/// Detect and Decode all types of Aztec Codes in images
pub struct AztecCodeDetector {
    width: usize,
    height: usize,
    image: Vec<bool>,
    markers: Vec<Marker>
}

impl AztecCodeDetector {

    /// Create a new AztecReader struct from a grayscale image
    /// 
    /// # Arguments
    /// * (`w`, `h`): The width and height of the grayscale image
    /// * `mono`: The grayscale pixel array (8 bits)
    pub fn from_grayscale((w, h): (u32, u32), mono: &[u8]) -> AztecCodeDetector {
        AztecCodeDetector {
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

            let (r, c) = (row + middle, col - middle + d - bl);
            sample[sc - 1 - i + 2 * sc] = self.sample_block(r, c, bl, bl);
            self.markers.push(Marker::blue((c, r), (bl, bl)));

            let (r, c) = (row - middle + d - bl, col - middle - bl);
            sample[sc - 1 - i + 3 * sc] = self.sample_block(r, c, bl, bl);
            self.markers.push(Marker::red((c, r), (bl, bl)));
        }

        sample
    }

    /// Extract the Aztec Code metadata including the number of layers and the
    /// number of codewords contained in the Overhead message. This function
    /// also distingushes Rune Aztec codes from Compact Aztec Codes.
    fn get_aztec_metadata(&self, mk: Marker, code_type: &mut AztecCodeType,
        message: &[bool]) -> Result<(usize, usize), AztecReadError> {

        let (len, block_size, count, mut corner) =
            if *code_type == AztecCodeType::FullSize {
                (56, 5u8, 10, Some(false))
            } else {
                (40, 7u8, 7, None)
            };

        if message.len() != len {
            return Err(AztecReadError::CorruptedFormat(mk,
                    "Invalid message size".to_owned()));
        }
        let mut codeword = 0;
        let mut codeword_size = 0u8;
        let mut block = 0;
        let mut msg = vec![0; count];
        let mut i = 1;
        let mut j = 0;
        while i < len {
            // extract the codewords by skipping the corners and anchor grid
            if block == block_size {
                i += match corner {
                    None | Some(true) => 3,
                    _ => 1
                };
                block = 0;
                corner = corner.map(|x| !x);
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

            if rs.fix_errors(&mut msg, 5).is_ok() {
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

    /// Tries to decode an Aztec Code candidate (AztecCenter) and retrive all
    /// its data. This may fail because of various resasons, please check the
    /// AztecReadError enum.
    pub fn decode(&mut self, center: AztecCenter)
        -> Result<ReadAztecCode, AztecReadError> {
        let (col, row) = center.loc;

        let (mut code_type, radius) =
            if self.check_ring(row, col, center.mod_size, 12.3) {
                (AztecCodeType::FullSize, 13.5)
            } else {
                (AztecCodeType::Compact, 9.5)
            };

        let sz = center.mod_size.round() as usize;
        let overhead_message = self.sample_ring(&center, radius, sz);
        let (layers, codewords) =
            self.get_aztec_metadata(center.as_marker(), &mut code_type,
            &overhead_message)?;

        if code_type == AztecCodeType::Rune {
            return Ok(ReadAztecCode { loc: center.loc, size: 11, code_type,
                data: vec![codewords as u8], center, codewords: 1, layers: 0,
                features: vec![] });
        }

        let compact = code_type == AztecCodeType::Compact;
        let (size, md2, samples) = if compact { // md2 <=> (bullseye size) / 2
            (11 + 4 * layers, 5, layers * 2)
        } else {
            let raw_size = 15 + 4 * layers;
            let anchor_grid = ((raw_size - 1) / 2 - 1) / 15;
            let size = raw_size + 2 * anchor_grid;
            (size, 7, (layers + anchor_grid) * 2)
        };

        let mut copy = AztecCode::new(compact, size);
        let mid = size / 2;
        for l in 1..=samples {
            let ring = self.sample_ring(&center, radius + 2.0 * l as f32, sz);
            let arc = ring.len() / 4; // arc length
            for cursor in 0..arc {
                copy[(mid - md2 - l, mid - md2 + 1 + cursor - l)] =
                    ring[cursor];

                copy[(mid - md2 + 1 + cursor - l, mid + md2 + l)] =
                    ring[cursor + arc];

                copy[(mid + md2 + l, mid - md2 + cursor - l)] =
                    ring[3 * arc - 1 - cursor];

                copy[(mid - md2 + cursor - l, mid - md2 - l)] =
                    ring[4 * arc - 1 - cursor];
            }
        }

        let mut rd = AztecReader::new(codewords, layers);
        rd.extract(copy);
        let (data, features) = rd.read().map_err(|msg|
            AztecReadError::CorruptedMessage(center.as_marker(), msg))?;

        Ok(ReadAztecCode { loc: center.loc, size, data, features,
            code_type, center, layers, codewords })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reader_rm_bit_stuffing() {
        let inp = "0010011100100000011010011011110000101001111001010000010110";
        let exp = "00100111001000000101001101111000010100111100101000000110";
        let mut bitstr: Vec<bool> = inp.chars().map(|x| x == '1').collect();
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        AztecReader::remove_bitstuffing(&mut bitstr, 6);
        assert_eq!(expected, bitstr);
    }
}
