//! Aztec Writer module
//!
//! Raztec supports generating any type of Aztec codes from compact (1-4
//! layers) to full-size (5-32 layers) and runes (1 byte). This module contains
//! all the necessary tools to fully generate Aztec Codes. Aztec Codes encoding
//! and generation and implemented according to the official Aztec Code Barcode
//! Symbology Specification (ISO/IEC 24778).
//!
//! Aztec Codes can contain different types of information as Text, Binary data
//! or more complex text using Enhanced Channel Interpretation (ECI). Pleas
//! keep in mind that Aztec Codes containing ECI must be read by compatible
//! scanners (at the risk of data corruption).
//!
//! # Examples
//!
//! Here is an example of generating an Aztec Code containing strings and
//! bytes:
//! ```rust
//!use raztec::writer::AztecCodeBuilder;
//!let code = AztecCodeBuilder::new().error_correction(50)
//!    .append("Hello").append(", ").append_bytes("World!".as_bytes())
//!    .build().unwrap();
//! ```
//!
//! Here is an example of generating an Aztec rune with the value 38:
//! ```rust
//!use raztec::writer::build_rune;
//!let code = build_rune(38);
//! ```
use super::{*, reed_solomon::ReedSolomon};

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

struct AztecWriter {
    size: usize,
    layers: usize,
    compact: bool,
    dominos: Vec<Domino>,
    codewords: usize,
    current_domino: usize,
    current_bit: bool
}

impl AztecWriter {
    fn new(codewords: usize, layers: usize, start_align: usize) -> Self {
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

        AztecWriter {
            size, layers, dominos, codewords, current_domino: start_align,
            current_bit: false, compact
        }
    }

    fn fill(&mut self, bitstr: &[bool]) {
        let mut idx = self.current_domino;
        for &bit in bitstr {
            let mut domino = &mut self.dominos[idx];
            if self.current_bit {
                domino.tail = bit;
                idx += 1;
            } else {
                domino.head = bit;
            }
            self.current_bit = !self.current_bit;
        }
        self.current_domino = idx;
    }

    fn into_aztec(self) -> AztecCode {
        let mut code = AztecCode::new(self.compact, self.size);

        for domino in &self.dominos {
            if domino.dir == Direction::None {
                continue;
            }
            code[domino.head()] = domino.head;
            code[domino.tail()] = domino.tail;
        }

        let layers = self.layers - 1;
        let words = self.codewords - 1;
        let rs = ReedSolomon::new(4, 0b10011);

        let (words, data) =
            if self.compact {
                let mut data = vec![
                    (layers << 2) | ((words >> 4) & 3),
                    words & 15
                ];
                data.extend(rs.generate_check_codes(&data, 5));
                (28, data)
            } else {
                let mut data = vec![
                    layers >> 1,
                    ((words & 0b11100000000) >> 8) | (layers & 1) << 3,
                    (words & 0b00011110000) >> 4,
                    words & 0b00000001111,
                ];
                data.extend(rs.generate_check_codes(&data, 6));
                (40, data)
            };

        let mut service_message = vec![false; words];
        let mut i = 0;
        for b in data.iter() {
            for j in 0..4 {
                service_message[i + 3 - j] = (b >> j) & 1 == 1;
            }
            i += 4;
        }

        if self.compact {
            Self::fill_compact_service_message(self.size / 2,
                &service_message, &mut code);
        } else {
            Self::fill_full_service_message(self.size / 2,
                &service_message, &mut code);
        }

        code
    }

    fn fill_compact_service_message(middle: usize, service_message: &[bool],
        code: &mut AztecCode) {
        let start_idx = middle - 5;
        for i in 0..7 {
            code[(start_idx,  start_idx + 2 + i)] = service_message[i     ];
            code[(start_idx + 2 + i, middle + 5)] = service_message[i +  7];
            code[(middle + 5,    middle + 3 - i)] = service_message[i + 14];
            code[(middle + 3 - i,     start_idx)] = service_message[i + 21];
        }
    }

    fn fill_full_service_message(middle: usize, service_message: &[bool],
        code: &mut AztecCode) {
        let start_idx = middle - 7;
        let end_idx = middle + 7;
        for i in 0..5 {
            code[(start_idx, start_idx + 2 + i)] = service_message[i     ];
            code[(start_idx, start_idx + 8 + i)] = service_message[i +  5];
            code[(start_idx + 2 + i,   end_idx)] = service_message[i + 10];
            code[(start_idx + 8 + i,   end_idx)] = service_message[i + 15];
            code[(end_idx,      middle + 5 - i)] = service_message[i + 20];
            code[(end_idx,      middle - 1 - i)] = service_message[i + 25];
            code[(middle + 5 - i,    start_idx)] = service_message[i + 30];
            code[(middle - 1 - i,    start_idx)] = service_message[i + 35];
        }
    }
}

impl Display for AztecWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut blocks = vec![(0usize, "##"); self.size * self.size];
        let mut code = AztecCode::new(self.compact, self.size);

        for (i, domino) in self.dominos.iter().enumerate() {
            if domino.dir == Direction::None {
                writeln!(f, "found empty domino at idx {}", i)?;
                continue;
            }
            let (row1, col1) = domino.head();
            let (row2, col2) = domino.tail();
            code[(row1, col1)] = domino.head;
            code[(row2, col2)] = domino.tail;
            let idx1 = row1 * self.size + col1;
            let color = i % 8;
            let (_, current) = blocks[idx1];
            if current != "##" {
                writeln!(f, "head collision detected")?;
                blocks[row2 * self.size + col2] = (color, "@@");
                continue;
            }
            let (_, current) = blocks[row2 * self.size + col2];
            if current != "##" {
                writeln!(f, "tail collision detected")?;
                blocks[row2 * self.size + col2] = (color, "$$");
                continue;
            }
            match domino.dir {
                Direction::Right => blocks[idx1] = (color, "->"),
                Direction::Up    => blocks[idx1] = (color, "^^"),
                Direction::Left  => blocks[idx1] = (color, "<-"),
                Direction::Down  => blocks[idx1] = (color, "vv"),
                Direction::None  => unreachable!()
            }
            blocks[row2 * self.size + col2] = (color, "██");
        }

        for row in 0..self.size {
            for col in 0..self.size {
                let (color, chars) = blocks[row * code.size + col];
                write!(f, "\x1b[9{}m{}", color, chars)?;
            }
            writeln!(f, "\x1b[0m")?;
        }
        Ok(())
    }
}

const LATCH_TABLE: [usize; 25] = [
// From  | To:  Upper      Lower      Mixed              Punct       Digit
/* Upper */     32767,        28,        29,        (30<<5)|29,         30,
/* Lower */(29<<5)|29,     32767,        29,        (30<<5)|29,         30,
/* Mixed */        29,        28,     32767,                30, (30<<5)|29,
/* Punct */        31,(28<<5)|31,(29<<5)|31,             32767, (30<<5)|31,
/* Digit */        14,(28<<4)|14,(29<<4)|14,(30<<9)|(29<<4)|14,      32767,
];

const SHIFT_TABLE: [usize; 25] = [
// From  | To:   Upper      Lower      Mixed   Punct     Digit
/* Upper */     32767,     32767,     32767,     0,      32767,
/* Lower */        28,     32767,     32767,     0,      32767,
/* Mixed */     32767,     32767,     32767,     0,      32767,
/* Punct */     32767,     32767,     32767, 32767,      32767,
/* Digit */        15,     32767,     32767,     0,      32767,
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Upper,
    Lower,
    Mixed,
    Punctuation,
    Digit
}

impl Mode {
    fn val(&self) -> usize {
        match self {
            Mode::Upper => 0,
            Mode::Lower => 1,
            Mode::Mixed => 2,
            Mode::Punctuation => 3,
            Mode::Digit => 4,
        }
    }

    fn capacity(&self) -> (usize, usize) {
        match self {
            Mode::Digit => (15, 4),
            _ => (31, 5)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Word {
    Char(u8),
    Punc(u8),
    Mixed(u8),
    Digit(u8),
    Byte(u8),
    Size(u16),
    Flg(u8)
}

impl Word {
    fn new(kind: Mode, c: u8) -> Self {
        match kind {
            Mode::Upper | Mode::Lower | Mode::Mixed => Word::Char(c),
            Mode::Digit                             => Word::Digit(c),
            Mode::Punctuation                       => Word::Punc(c)
        }
    }

    fn upper_letter(c: char) -> Self {
        Word::Char(c as u8 - 65 + 2)
    }

    fn lower_letter(c: char) -> Self {
        Word::Char(c as u8 - 97 + 2)
    }

    fn digit(c: char) -> Self {
        Word::Digit(c as u8 - 48 + 2)
    }
}

/// Aztec Code config containing all the different properties of an Aztec Code
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AztecCodeConfig {
    layers: usize,
    bit_cap: usize,
    words: usize,
    codewords: usize,
    codeword_size: usize,
    check_codes: usize
}

impl AztecCodeConfig {

    /// Number of layers of the Aztec Code (1-32)
    pub fn layers(&self) -> usize { self.layers }

    /// Capacity in bits of the Aztec Code including the error check codewords.
    pub fn total_capacity(&self) -> usize { self.bit_cap }

    /// Number of words appended to the Aztec Code
    pub fn words(&self) -> usize { self.words }

    /// Number of codewords encoded in the Aztec Code
    pub fn nb_codewords(&self) -> usize { self.codewords }

    /// Size of a codeword in bits (6-12)
    pub fn codeword_size(&self) -> usize { self.codeword_size }

    /// Number of error correction codes appended to the end of the Aztec Code
    pub fn nb_check_codes(&self) -> usize { self.check_codes }
}

/// Aztec Code generator using the Builder Pattern
pub struct AztecCodeBuilder {
    current_mode: Mode,
    words: Vec<Word>,
    ecr: usize // Error Correction Rate
}

impl AztecCodeBuilder {

    /// Creates an AztecCodeBuilder with a default error correction rate of 23%
    pub fn new() -> AztecCodeBuilder {
        AztecCodeBuilder {
            current_mode: Mode::Upper, words: Vec::new(), ecr: 23
        }
    }

    /// Sets the error correction rate percentage of the final Aztec Code.
    /// Minimim correction rate is 5% and Maximum is rate is 95%.
    pub fn error_correction(&mut self, rate: usize) -> &mut AztecCodeBuilder {
        assert!((5..96).contains(&rate), "Invalid error correction rate 5-95");
        self.ecr = rate;
        self
    }

    /// Appends byte slice to the Aztec Code according to the Aztec Code
    /// Specification. This method only accepts byte slices up to 2079 bytes
    /// (31 + 2^11). Note that if the length of the byte slice is between 32
    /// and 62 bytes, two binary blocks are generated instead of a big one as
    /// it is more space efficient.
    pub fn append_bytes(&mut self, bytes: &[u8]) -> &mut AztecCodeBuilder {
        let len = bytes.len();
        if len == 0 {
            return self;
        }

        if (32..=62).contains(&len) {
            // Two 5-bit byte shift sequences are more compact than one 11-bit.
            self.append_bytes(&bytes[0..31]);
            return self.append_bytes(&bytes[31..]);
        }

        // shift to Binary mode
        if self.current_mode == Mode::Punctuation {
            self.words.push(Word::Char(31)); // switch to Upper Mode (no B/S)
            self.current_mode = Mode::Upper;
        }
        self.words.push(Word::Char(31));

        if (1..=31).contains(&len) {
            self.words.push(Word::Char(len as u8));
        } else {
            self.words.push(Word::Char(0));
            self.words.push(Word::Size((len - 31) as u16));
        }
        self.words.extend(bytes.iter().map(|&x| Word::Byte(x)));
        self
    }

    /// Appends an ECI escape code to the Aztec Code.
    /// The default ECI indicator is \000003 (code = 3).
    pub fn append_eci(&mut self, code: u16) -> &mut AztecCodeBuilder {
        let repr = code.to_string();
        assert!(repr.len() < 7, "ECI codes with 7+ digits are illegal");
        self.push_in((Word::Flg(repr.len() as u8), Mode::Punctuation), None);
        self.words.extend(repr.chars().map(Word::digit));
        self
    }

    /// Appends text and convert it to the Aztec Code format to the future
    /// Aztec Code. Note that the size of the final Aztec Code is not known
    /// until the `build` function is called. Panics if there is not supported
    /// characters in the string (Characters must be in ASCII-128).
    pub fn append(&mut self, text: &str) -> &mut AztecCodeBuilder {
        self.try_append(text).unwrap()
    }

    /// Tries to append text and convert it to the Aztec Code format to the
    /// future Aztec Code. Note that the size of the final Aztec Code is not
    /// known until the `build` function is called. Returns an error if there
    /// is not supported characters in the string (Characters must be in
    /// ASCII-128).
    pub fn try_append(&mut self, text: &str)
        -> Result<&mut AztecCodeBuilder, String> {
        if text.is_empty() {
            return Ok(self);
        }
        let mut chars = text.chars();
        let mut prev_word = self.process_char(chars.next().unwrap(), 0)?;
        for (i, c) in chars.enumerate() {
            prev_word = match (c, prev_word) {
                // Check for 2-bytes words combinations
                ('\n', (Word::Punc(1), Mode::Punctuation))
                    => (Word::Punc(2), Mode::Punctuation),

                (' ',  (Word::Punc(19), Mode::Punctuation))
                    => (Word::Punc(3),  Mode::Punctuation),

                (' ',  (Word::Punc(17), Mode::Punctuation))
                    => (Word::Punc(4),  Mode::Punctuation),

                (' ',  (Word::Punc(21), Mode::Punctuation))
                    => (Word::Punc(5),  Mode::Punctuation),

                _ => {
                    let next = self.process_char(c, i + 1)?;
                    self.push_in(prev_word, Some(next));
                    next
                }
            }
        }
        self.push_in(prev_word, None);
        Ok(self)
    }

    /// Returns the next state according to the current mode and the modes in
    /// which the character can be encoded (assuming this function is called
    /// when processing a character).
    fn poly_word(&self, accepted_mode: Mode, original: Word,
        normal_mode: Mode, other: Word) -> (Word, Mode) {
        if self.current_mode != accepted_mode {
            (other, normal_mode)
        } else {
            (original, accepted_mode)
        }
    }

    /// Converts the provided char into its translation in the Aztec code
    /// Character set. This convertion takes into account the current mode of
    /// the builder.
    fn process_char(&self, c: char, i: usize) -> Result<(Word, Mode), String> {
        match c as u32 {
            1..=12 => Ok((Word::Mixed(c as u8 + 1), Mode::Mixed)),
            13 => Ok(self.poly_word(Mode::Mixed, Word::Mixed(14), // \r
                                 Mode::Punctuation, Word::Punc(1))),
            27..=31 => Ok((Word::Mixed(c as u8 - 27 + 15), Mode::Mixed)),
            32 => { // space
                if self.current_mode != Mode::Punctuation {
                    Ok((Word::new(self.current_mode, 1), self.current_mode))
                } else {
                    Ok((Word::Char(1), Mode::Upper))
                }
            },
            44 => Ok(self.poly_word(Mode::Digit, Word::Digit(12), // ,
                                 Mode::Punctuation, Word::Punc(17))),
            46 => Ok(self.poly_word(Mode::Digit, Word::Digit(13), // .
                                 Mode::Punctuation, Word::Punc(19))),
            33..=47   => Ok((Word::Punc(c as u8 - 33 + 6), Mode::Punctuation)), // ! -> /
            48..=57   => Ok((Word::digit(c), Mode::Digit)),
            58..=63   => Ok((Word::Punc(c as u8 - 58 + 21), Mode::Punctuation)), // : -> ?
            64        => Ok((Word::Mixed(20), Mode::Mixed)), // @
            65..=90   => Ok((Word::upper_letter(c), Mode::Upper)),
            91        => Ok((Word::Punc(27), Mode::Punctuation)), // [
            92        => Ok((Word::Mixed(21), Mode::Mixed)), // \
            93        => Ok((Word::Punc(28), Mode::Punctuation)), // ]
            94..=96   => Ok((Word::Mixed(c as u8 - 94 + 22), Mode::Mixed)), // ^ -> `
            97..=122  => Ok((Word::lower_letter(c), Mode::Lower)),
            123       => Ok((Word::Punc(29), Mode::Punctuation)), // {
            124       => Ok((Word::Mixed(25), Mode::Mixed)), // |
            125       => Ok((Word::Punc(30), Mode::Punctuation)), // } 
            126..=127 => Ok((Word::Mixed(c as u8 - 100), Mode::Mixed)), // ~ -> DEL
            x => panic!("Character not supported `{}` (code: {}) at index {}",
                c.escape_default(), x, i)
        }
    }

    /// Pushes the couple of (Word, Mode) generated by the process_char
    /// function. All the mode switching logic is done here. The new mode is
    /// generated using the current mode of the builder, the expected_mode and
    /// the mode of the next (Word, Mode) couple (if present).
    fn push_in(&mut self, (word, expected_mode): (Word, Mode),
        next: Option<(Word, Mode)>) {
        let mut cur_mode = self.current_mode;

        if cur_mode != expected_mode {
            // get the combination of words to switch from current to next mode
            let switch = cur_mode.val() * 5 + expected_mode.val();
            let mut code = LATCH_TABLE[switch];

            self.current_mode = expected_mode;
            let (_, next_mode) = next.unwrap_or((word, cur_mode));
            // see if we can do a shift instead of a latch
            if next_mode == cur_mode {
                let shift = SHIFT_TABLE[switch];
                if shift != 32767 {
                    code = shift;
                    // go back to the last mode immediatly
                    self.current_mode = cur_mode;
                }
            }

            let (mut limit, mut shift) = cur_mode.capacity();
            while code > limit { // at most 2 words are needed to change mode
                self.words.push(Word::new(cur_mode, (code & limit) as u8));
                code >>= shift;
                cur_mode = Mode::Upper; // force next word's bit len to 5 bits
                (limit, shift) = cur_mode.capacity();
            }
            self.words.push(Word::new(cur_mode, code as u8));
        }
        self.words.push(word);
    }

    /// Copies `bits` bits starting from the MSB from `byte` to
    /// the end of bit string. Note that `bits` should not exeed 8 (max 1b).
    fn append_bits(&self, bitstr: &mut Vec<bool>, byte: u8, bits: u8)  {
        bitstr.extend((0..bits).rev().map(|bit| ((byte >> bit) & 1) == 1));
    }

    /// Converts and concatenates all words to a bit string using their binary
    /// representation following the Aztec Spec. Words take 4 bits while in
    /// Digit mode, 8 bits in Binary Mode and 5 bits in the other ones.
    fn to_bit_string(&self) -> Vec<bool> {
        let mut bitstr = Vec::new();

        for &word in self.words.iter() {
            match word {
                Word::Size(len) => {
                    self.append_bits(&mut bitstr, (len >>  8) as u8, 3);
                    self.append_bits(&mut bitstr, (len & 255) as u8, 8)
                },
                Word::Flg(n) => {
                    self.append_bits(&mut bitstr, 0, 5);
                    self.append_bits(&mut bitstr, n, 3)
                },
                Word::Byte(x)  => self.append_bits(&mut bitstr, x, 8),
                Word::Digit(x) => self.append_bits(&mut bitstr, x, 4),
                Word::Char(x) | Word::Punc(x) | Word::Mixed(x) 
                    => self.append_bits(&mut bitstr, x, 5)
            }
        }
        bitstr
    }

    /// Searches codewords that start with the same 5 consecutive bitstr
    /// and inserts the complementary value after them.
    /// Example: `00000x$` and `11111x$` 
    ///           where `$` can be a 0 or 1 and `x` the inserted bit.
    fn bit_stuffing(&self, bitstr: &mut Vec<bool>, codeword_size: usize) {
        if bitstr.len() < codeword_size {
            return;
        }
        let mut i = 0;
        let l = bitstr.len() - codeword_size;
        let limit = codeword_size - 1;
        while i < l {
            let first = bitstr[i];
            let mut j = 1;
            while j < limit && bitstr[i + j] == first {
                j += 1;
            }
            if j == limit {
                bitstr.insert(i + limit, !first);
            }
            i += codeword_size;
        }
    }

    /// Add padding bits to the bit string to make its size a multiple of
    /// codeword size. Padding bits are 1s unless the last codeword forms a 
    /// codeword full of ones. In that case, the last padding bit is a zero
    /// instead of a one (bit stuffing).
    fn add_padding(&self, bitstr: &mut Vec<bool>, codeword_size: usize) {
        let remaining = bitstr.len() % codeword_size;
        if remaining == 0 {
            return;
        }

        // pad the last bits by 1s to next codeword boundary
        bitstr.extend(std::iter::repeat(true).take(codeword_size - remaining));

        // check if the last codeword is all ones to do one bit stuffing
        let len = bitstr.len();
        let mut i = len - codeword_size;
        let limit = i + remaining;
        while i < limit && bitstr[i] {
            i += 1;
        }
        if i == limit {
            bitstr[len - 1] = false;
        }
    }

    /// Find the number of layers needed to fit `total_bits` bits. Returns
    /// (layers, total_bits) where total_bits is the total capacity of all the
    /// layers combined.
    pub fn find_nb_layers(total_bits: usize) -> (usize, usize) {
        let mut layers = 1;
        let mut nb_bits = (88 + 16 * layers) * layers;

        while nb_bits < total_bits && layers < 4 {
            layers += 1;
            nb_bits = (88 + 16 * layers) * layers;
        }

        if layers == 4 && nb_bits < total_bits {
            // if we couldn't fit `total_bits` in a compact aztec code, try
            // with a full-size Aztec code (starting with 5 layers)
            layers += 1;
            nb_bits = (112 + 16 * layers) * layers;
            while nb_bits < total_bits {
                layers += 1;
                nb_bits = (112 + 16 * layers) * layers;
            }
        }
        (layers, nb_bits)
    }

    /// Converts the final bit string back into codewords to generate Reed
    /// Solomon check codewords.
    fn to_words(&self, bitstr: &[bool], size: usize) -> Vec<usize> {
        let l = bitstr.len();
        // we know that the bitstr len is exactly a multiple of the codeword 
        // size thanks to add_padding
        let mut bytes = Vec::with_capacity(l / size);
        for i in (0..l).step_by(size) {
            let mut val = 0;
            for j in 0..size {
                val = (val << 1) | bitstr[i + j] as usize;
            }
            bytes.push(val);
        }
        bytes
    }

    /// Simulates the generation of an Aztec Code and returns its final
    /// configuration. This operation may fail as the build function.
    pub fn build_config(&self) -> Result<AztecCodeConfig, String> {
        let mut bitstr = self.to_bit_string();

        let (layers, bits_in_layers) =
            Self::find_nb_layers(bitstr.len() + bitstr.len() * self.ecr / 100);
        let codeword_size = match layers {
             1..=2  =>  6,
             3..=8  =>  8,
             9..=22 => 10,
            23..=32 => 12,
            _ => return Err(
                format!("Aztec code with {} layers is not supported", layers))
        };

        self.bit_stuffing(&mut bitstr, codeword_size);
        self.add_padding(&mut bitstr, codeword_size);

        let codewords = bitstr.len() / codeword_size;
        let check_codes = (bits_in_layers - bitstr.len()) / codeword_size;

        Ok(AztecCodeConfig { layers, bit_cap: bits_in_layers, codewords,
            words: self.words.len(), codeword_size, check_codes })
    }

    /// Generates an AztecCode from the current state of the builder. The state
    /// of the builder can be changed using `append` functions. Returns an
    /// error if the builder's content can not fit in a valid Aztec Code.
    pub fn build(&self) -> Result<AztecCode, String> {
        let mut bitstr = self.to_bit_string();

        // Reed Solomon Config
        let (layers, bits_in_layers) =
            Self::find_nb_layers(bitstr.len() + bitstr.len() * self.ecr / 100);
        let (codeword_size, prim) = match layers {
             1..=2  => ( 6,       0b1000011),
             3..=8  => ( 8,     0b100101101),
             9..=22 => (10,   0b10000001001),
            23..=32 => (12, 0b1000001101001),
            _ => return Err(
                format!("Aztec code with {} layers is not supported", layers))
        };

        self.bit_stuffing(&mut bitstr, codeword_size);
        self.add_padding(&mut bitstr, codeword_size);
        let words = self.to_words(&bitstr, codeword_size);

        let codewords = bitstr.len() / codeword_size;
        let to_fill = (bits_in_layers - bitstr.len()) / codeword_size;
        let rs = ReedSolomon::new(codeword_size as u8, prim);

        let check_words = rs.generate_check_codes(&words, to_fill);
        if codeword_size > 8 {
            for check_word in check_words {
                self.append_bits(&mut bitstr, (check_word >> 8) as u8,
                    (codeword_size - 8) as u8);
                self.append_bits(&mut bitstr, (check_word & 0xFF) as u8, 8);
            }
        } else {
            for check_word in check_words {
                self.append_bits(&mut bitstr, check_word as u8,
                    codeword_size as u8);
            }
        }

        let start_align = (bits_in_layers % codeword_size) / 2;
        let mut writer = AztecWriter::new(codewords, layers, start_align);
        writer.fill(&bitstr);
        Ok(writer.into_aztec())
    }
}

impl Default for AztecCodeBuilder {
    fn default() -> Self {
        AztecCodeBuilder::new()
    }
}

/// Generate an Aztec Code Rune containing a single byte of information
pub fn build_rune(data: u8) -> AztecCode {
    let mut code = AztecCode::new(true, 11);

    let rs = ReedSolomon::new(4, 0b10011);
    let mut data = vec![
        ((data >> 4) & 0b1111) as usize,
        (data & 0b1111) as usize
    ];
    data.extend(rs.generate_check_codes(&data, 5));

    let mut service_message = vec![false; 28];
    let mut i = 0;
    for b in data.iter() {
        // in order to distinguish them from normal overhead messages,
        // each bit is inverted at the graphical level (word xor 1010)
        let b = b ^ 0b1010;
        for j in 0..4 {
            service_message[i + 3 - j] = (b >> j) & 1 == 1;
        }
        i += 4;
    }

    AztecWriter::fill_compact_service_message(11 / 2,
        &service_message, &mut code);
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_padding() {
        let inp = "00100111";
        let exp = "001001111110";
        let mut bitstr: Vec<bool> = inp.chars().map(|x| x == '1').collect();
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        AztecCodeBuilder::new().add_padding(&mut bitstr, 6);
        assert_eq!(expected, bitstr);
    }

    #[test]
    fn test_add_padding2() {
        let inp = "001001101";
        let exp = "001001101111";
        let mut bitstr: Vec<bool> = inp.chars().map(|x| x == '1').collect();
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        AztecCodeBuilder::new().add_padding(&mut bitstr, 6);
        assert_eq!(expected, bitstr);
    }

    #[test]
    fn test_bit_stuffing() {
        let inp = "00100111001000000101001101111000010100111100101000000110";
        let exp = "0010011100100000011010011011110000101001111001010000010110";
        let mut bitstr: Vec<bool> = inp.chars().map(|x| x == '1').collect();
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        AztecCodeBuilder::new().bit_stuffing(&mut bitstr, 6);
        assert_eq!(expected, bitstr);
    }

    #[test]
    fn test_to_bitstr() {
        use Word::*;
        let inp = vec![Char(8), Char(28), Char(16), Char(0), Punc(6)];
        let exp = "0100011100100000000000110";
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        let mut builder = AztecCodeBuilder::new();
        builder.words = inp;
        assert_eq!(expected, builder.to_bit_string());
    }

    #[test]
    fn test_to_bitstr2() {
        use Word::*;
        let inp = vec![Char(14), Char(30), Digit(3), Digit(6)];
        let exp = "011101111000110110";
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        let mut builder = AztecCodeBuilder::new();
        builder.words = inp;
        assert_eq!(expected, builder.to_bit_string());
    }

    #[test]
    fn test_append_bytes() {
        let inp = vec![0x47, 0x6f, 0x74, 0x6f, 0x75, 0x62, 0x75, 0x6e];
        let exp =
"11111010000100011101101111011101000110111101110101011000100111010101101110";
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        let mut builder = AztecCodeBuilder::new();
        builder.append_bytes(&inp);
        assert_eq!(expected, builder.to_bit_string());
    }

    #[test]
    fn test_append_eci() {
        let exp = "11111000011011011000000000000011001111110000110110110";
        let expected: Vec<bool> = exp.chars().map(|x| x == '1').collect();
        let mut builder = AztecCodeBuilder::new();
        builder
            .append_bytes(&[182])
            .append_eci(7)
            .append_bytes(&[182]);
        assert_eq!(expected, builder.to_bit_string());
    }
}
