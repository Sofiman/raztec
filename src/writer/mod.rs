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
            1..=2 => 6,
            3..=8 => 8,
            9..=22 => 10,
            23..=32 => 12,
            _ => panic!("Too many layers")
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

        AztecWriter { size, dominos, codeword_size, current_domino: 0 }
    }

    pub fn into_aztec(self) -> AztecCode {
        let mut code = AztecCode::new(self.size);

        for domino in &self.dominos {
            code[domino.head()] = domino.head;
            code[domino.tail()] = domino.tail;
        }

        code
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
        let nb_dominos = buf.len() * 8 / 4;
        let remaining = self.dominos.len() - self.current_domino;

        if remaining < nb_dominos {
            return Ok(0)
        }

        let mut idx = self.current_domino;
        for byte in buf {
            for bit in (0..8).step_by(2) {
                let mut domino = &mut self.dominos[idx];
                domino.head = ((byte >> bit)       & 1) == 1;
                domino.tail = ((byte >> (bit + 1)) & 1) == 1;
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
        let mut blocks = vec![(0usize, "██"); self.dominos.len()*4];

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
