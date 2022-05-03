use super::{*, reed_solomon::ReedSolomonEncoder};

#[derive(Debug)]
struct Domino {
    dir: Direction,
    head_pos: (usize, usize),
    head: bool,
    tail: bool
}

#[derive(Debug)]
enum Direction {
    Down,
    Left,
    Up,
    Right,
}

impl Domino {
    fn down(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Down, head: false, tail: false }
    }

    fn left(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Left, head: false, tail: false }
    }

    fn up(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Up, head: false, tail: false }
    }

    fn right(head_pos: (usize, usize)) -> Self {
        Self { head_pos, dir: Direction::Right, head: false, tail: false }
    }

    fn head(&self) -> (usize, usize) {
        self.head_pos
    }

    fn tail(&self) -> (usize, usize) {
        let (row, col) = self.head_pos;
        match &self.dir {
            Direction::Down  => (row + 1, col),
            Direction::Left  => (row, col - 1),
            Direction::Up    => (row - 1, col),
            Direction::Right => (row, col + 1)
        }
    }
}

struct AztecWriter {
    size: usize,
    dominos: Vec<Domino>,
    codewords: usize,
    current_domino: usize,
    current_bit: bool
}

impl AztecWriter {
    fn new(codewords: usize, layers: usize) -> Self {
        let size = layers * 4 + 11;
        let mut dominos = Vec::new();

        for layer in 0..layers {
            let start = 2 * layer;
            let end = size - start - 1;
            let limit = end - start - 1;

            for row in 0..limit {
                dominos.push(Domino::right((start + row, start)))
            }

            for col in 0..limit {
                dominos.push(Domino::up((end, start + col)))
            }

            for row in 0..limit {
                dominos.push(Domino::left((end - row, end)))
            }

            for col in 0..limit {
                dominos.push(Domino::down((start, end - col)))
            }

        }

        let start = if layers == 1 { 1 } else { 0 };
        AztecWriter { 
            size, dominos, codewords, current_domino: start, current_bit: false
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
        let mut code = AztecCode::new(self.size);

        for domino in &self.dominos {
            code[domino.head()] = domino.head;
            code[domino.tail()] = domino.tail;
        }

        let mut service_message = [false; 28];
        let layers: u8 = ((self.size - 11) / 4 - 1) as u8;
        let words = self.codewords as u8 - 1;
        let data = [(layers << 2) | ((words >> 4) & 3), words & 15];

        /* reed_solomon */
        let encoder = ReedSolomonEncoder::new(4, 0b10011);
        let check_codes = encoder.generate_check_codes(&data, 5);
        let mut data = data.to_vec();
        data.extend(&check_codes);

        let mut i = 0;
        for b in data.iter() {
            for j in 0..4 {
                service_message[i + 3 - j] = (b >> j) & 1 == 1;
            }
            i += 4;
        }

        let middle = self.size / 2;
        let start_idx = middle - 5;
        for i in 0..7 {
            code[(start_idx, start_idx + 2 + i)]  = service_message[i     ];
            code[(start_idx + 2 + i, middle + 5)] = service_message[i +  7];
            code[(middle + 5, middle + 3 - i)]    = service_message[i + 14];
            code[(middle + 3 - i, start_idx)]     = service_message[i + 21];
        }

        code
    }
}

impl Display for AztecWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut blocks = vec![(0usize, "██"); self.size * self.size];

        let mut code = AztecCode::new(self.size);

        for (i, domino) in self.dominos.iter().enumerate() {
            let (row1, col1) = domino.head();
            let (row2, col2) = domino.tail();
            code[(row1, col1)] = domino.head;
            code[(row2, col2)] = domino.tail;
            let idx1 = row1 * self.size + col1;
            let color = i % 8;
            match domino.dir {
                Direction::Right => blocks[idx1] = (color, "->"),
                Direction::Up    => blocks[idx1] = (color, "^^"),
                Direction::Left  => blocks[idx1] = (color, "<-"),
                Direction::Down  => blocks[idx1] = (color, "vv"),
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

const SWITCH_TABLE: [usize; 25] = [
// From  | To:   Upper      Lower      Mixed   Punct     Digit
/* Upper */       255,        28,        29,     0,         30,
/* Lower */(29<<5)|29,       255,        29,     0,         30,
/* Mixed */        29,        28,       255,     0, (30<<5)|29,
/* Punct */        31,(28<<5)|31,(29<<5)|31,   255, (30<<5)|31,
/* Digit */        14,(28<<4)|14,(29<<4)|14, 14<<4,        255,
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Upper,
    Lower,
    Mixed,
    Punctuation,
    Digit,
    Binary
}

impl Mode {
    fn val(&self) -> usize {
        match self {
            Mode::Upper => 0,
            Mode::Lower => 1,
            Mode::Mixed => 2,
            Mode::Punctuation => 3,
            Mode::Digit => 4,
            Mode::Binary => 5
        }
    }

    fn capacity(&self) -> (usize, usize) {
        match self {
            Mode::Digit => (15, 4),
            Mode::Binary => (255, 8),
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
    Byte(u8)
}

impl Word {
    fn new(kind: Mode, c: u8) -> Self {
        match kind {
            Mode::Upper | Mode::Lower => Word::Char(c),
            Mode::Mixed => Word::Char(c),
            Mode::Digit => Word::Digit(c),
            Mode::Punctuation => Word::Punc(c),
            Mode::Binary => Word::Byte(c)
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


pub struct AztecCodeBuilder {
    current_mode: Mode,
    words: Vec<Word>,
    compression: usize
}

impl AztecCodeBuilder {

    pub fn new(compression: usize) -> AztecCodeBuilder {
        if !(0..100).contains(&compression) {
            panic!("Invalid compression percentage");
        }
        AztecCodeBuilder {
            current_mode: Mode::Upper, words: Vec::new(), compression
        }
    }

    pub fn append(&mut self, text: &str) -> &mut AztecCodeBuilder {
        let chars: Vec<char> = text.chars().collect();
        for c in chars {
            let (word, mode) = match c as u8 {
                65..=90 => (Word::upper_letter(c), Mode::Upper),
                97..=122 => (Word::lower_letter(c), Mode::Lower),
                48..=57 => (Word::digit(c), Mode::Digit),
                44 => { // ,
                    if self.current_mode != Mode::Digit {
                        (Word::Punc(17), Mode::Punctuation)
                    } else {
                        (Word::Digit(12), Mode::Digit)
                    }
                },
                46 => { // .
                    if self.current_mode != Mode::Digit {
                        (Word::Punc(19), Mode::Punctuation)
                    } else {
                        (Word::Digit(13), Mode::Digit)
                    }
                },
                33..=47 => (Word::Punc(c as u8 - 33 + 6), Mode::Punctuation), // ! -> /
                58..=63 => (Word::Punc(c as u8 - 58 + 21), Mode::Punctuation), // : -> ?
                91 => (Word::Punc(27), Mode::Punctuation), // [
                93 => (Word::Punc(28), Mode::Punctuation), // ]
                123 => (Word::Punc(29), Mode::Punctuation), // {
                125 => (Word::Punc(30), Mode::Punctuation), // } 
                64 => (Word::Mixed(20), Mode::Mixed), // @
                92 => (Word::Mixed(21), Mode::Mixed), // \
                94..=96 => (Word::Mixed(c as u8 - 94 + 22), Mode::Mixed), // ^ -> `
                126 => (Word::Mixed(26), Mode::Mixed), // ~
                10 => (Word::Punc(1), Mode::Punctuation), // \n
                32 => { // space
                    if self.current_mode != Mode::Punctuation {
                        (Word::new(self.current_mode, 1), self.current_mode)
                    } else {
                        (Word::Char(1), Mode::Upper)
                    }
                },
                _ => continue
            };
            self.push_in(word, mode);
        }
        println!("{:?}", self.words);
        self
    }

    fn push_in(&mut self, word: Word, expected_mode: Mode) {
        let cur_mode = self.current_mode;
        if cur_mode != expected_mode {
            // get the combination of words to switch from current to next mode
            let switch = cur_mode.val() * 5 + expected_mode.val();
            let mut code = SWITCH_TABLE[switch];

            let mut to_add = Vec::new();
            let (mut limit, mut shift) = cur_mode.capacity();
            let mut switch_mode = cur_mode;
            while code > limit { // allow for multiple switch words
                to_add.push(Word::new(switch_mode, (code & limit) as u8));
                switch_mode = Mode::Upper; // force Char mode (5 bits encoded)
                code >>= shift;
                (limit, shift) = switch_mode.capacity();
            }
            to_add.push(Word::new(switch_mode, code as u8));
            self.words.append(&mut to_add);

            if expected_mode != Mode::Punctuation {
                // there is no Punctuation latch so we never stay in that mode
                self.current_mode = expected_mode;
            }
        }
        self.words.push(word);
    }

    fn append_bits(&self, bitstr: &mut Vec<bool>, byte: u8, bits: u8)  {
        for bit in 0..bits {
            let bit = bits - 1 - bit;
            bitstr.push(((byte >> bit) & 1) == 1);
        }
    }

    fn to_bit_string(&self) -> Vec<bool> {
        let mut bitstr = Vec::new();

        for &word in self.words.iter() {
            match word {
                Word::Byte(x) => self.append_bits(&mut bitstr, x, 8),
                Word::Digit(x) => self.append_bits(&mut bitstr, x, 4),
                Word::Char(x) | Word::Punc(x) | Word::Mixed(x) 
                    => self.append_bits(&mut bitstr, x, 5),
            }
        }
        bitstr
    }

    fn bit_stuffing(&self, bitstr: &mut Vec<bool>, codeword_size: usize) {
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

    fn add_padding(&self, bitstr: &mut Vec<bool>, codeword_size: usize) {
        let remaining = bitstr.len() % codeword_size;
        if remaining == 0 {
            return;
        }
        let remaining = codeword_size - remaining;
        for _i in 0..remaining {
            bitstr.push(true);
        }
    }

    fn find_nb_layers(&self, total_bits: usize) -> (usize, usize) {
        let mut layers = 1;
        while (88 + 16 * layers) * layers < total_bits {
            layers += 1;
        }
        (layers, (88 + 16 * layers) * layers)
    }

    fn to_words(&self, bitstr: &[bool], size: usize) -> Vec<u8> {
        let l = bitstr.len();
        let mut bytes = Vec::with_capacity(l / size);
        for i in (0..=(l - size)).step_by(size) {
            let mut val = 0;
            for j in 0..size {
                val <<= 1;
                val |= bitstr[i + j] as u8;
            }
            bytes.push(val);
        }
        bytes
    }

    pub fn build(&self) -> AztecCode {
        let mut bitstr = self.to_bit_string();

        // reed_solomon config
        let (layers, bits_in_layers) = self.find_nb_layers(bitstr.len() + 
            bitstr.len() * self.compression / 100);
        let (codeword_size, prim) = match layers {
            1..=2 => (6, 0b1000011),
            3..=8 => (8, 0b100101101),
            //9..=22 => (10, 0b10000001001),
            //23..=32 => (12, 0b1000001101001),
            _ => panic!("Aztec code with {} layers is not supported", layers)
        };

        self.bit_stuffing(&mut bitstr, codeword_size);
        self.add_padding(&mut bitstr, codeword_size);
        let words = self.to_words(&bitstr, codeword_size);

        let codewords = bitstr.len() / codeword_size;
        let to_fill = (bits_in_layers - bitstr.len()) / codeword_size;
        let rs = ReedSolomonEncoder::new(codeword_size as u8, prim);

        let check_words = rs.generate_check_codes(&words, to_fill);
        for check_word in check_words {
            self.append_bits(&mut bitstr, check_word, codeword_size as u8);
        }

        let mut writer = AztecWriter::new(codewords, layers);
        writer.fill(&bitstr);
        writer.into_aztec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_stuffing() {
        let input    = "00100111001000000101001101111000010100111100101000000110";
        let expected = "0010011100100000011010011011110000101001111001010000010110";
        let mut bitstr:Vec<bool> = input.chars().map(|x| x == '1').collect();
        let builder = AztecCodeBuilder::new(23);
        builder.bit_stuffing(&mut bitstr, 6);

        let result = bitstr.iter().fold(String::new(), |acc, &x| acc + if x { "1" } else { "0" });
        assert_eq!(expected, result);
    }
}
