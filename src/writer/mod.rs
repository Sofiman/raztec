use std::io::{self, Write, Error, ErrorKind};

use crate::*;

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

pub struct AztecWriter {
    size: usize,
    dominos: Vec<Domino>,
    codeword_size: usize,
    current_domino: usize
}

impl AztecWriter {
    pub fn new(layers: usize) -> Self {
        let size = layers * 4 + 11;
        let mut dominos = Vec::new();

        let codeword_size: usize = match layers {
            1..=2 => 3,
            3..=8 => 4,
            //9..=22 => 5,
            //23..=32 => 6,
            _ => panic!("Aztec code with {} layers is not supported", layers)
        };

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

        AztecWriter { size, dominos, codeword_size, current_domino: 1 }
    }

    pub fn into_aztec(self) -> AztecCode {
        let mut code = AztecCode::new(self.size);

        for domino in &self.dominos {
            code[domino.head()] = domino.head;
            code[domino.tail()] = domino.tail;
        }

        let mut service_message = [false; 28];
        let count = (self.size - 11) / 4 - 1;
        let codeword_count = (self.current_domino - 1) / self.codeword_size - 1;
        service_message[0] = count & 2 == 2;
        service_message[1] = count & 1 == 1;
        for i in 0..6 {
            service_message[2 + 5 - i] = (codeword_count >> i) & 1 == 1;
        }
        //self.error_correction(&mut service_message, (count << 6) + codeword_count);

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

    fn error_correction(&self, code: &mut [bool], codewords: usize) {
        let middle = self.size / 2;
        let col = middle + 5;

        let control1 = 0b1010;
        let control2 = 0b1010;
        let control3 = 0b1010;
        let control4 = 0b1010;
        let control5 = 0b1010;

        todo!();
    }
}

impl Write for AztecWriter {

    fn write_fmt(&mut self, fmt: std::fmt::Arguments<'_>) -> io::Result<()> {
        self.write_all(fmt.to_string().as_bytes())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let result = self.write(buf);
        if let Ok(written) = result {
            if written == 0 {
                return Err(Error::new(ErrorKind::WriteZero, "Not enough space left"))
            }
        }
        Ok(())
    }

    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let nb_dominos = buf.len() * 8 / 2;
        let remaining = self.dominos.len() - self.current_domino;

        if remaining < nb_dominos {
            return Ok(0)
        }

        let mut idx = self.current_domino;
        for byte in buf {
            for bit in (0..8).step_by(2) {
                let bit = 6 - bit;
                let mut domino = &mut self.dominos[idx];
                domino.head = ((byte >> (bit + 1)) & 1) == 1;
                domino.tail = ((byte >> bit)       & 1) == 1;
                idx += 1;
            }
        }
        self.current_domino = idx;

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
       Ok(()) 
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
