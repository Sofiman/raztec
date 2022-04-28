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
    current_domino: usize
}

impl AztecWriter {
    pub fn new(aztec_code_size: usize) -> Self {
        let end = aztec_code_size - 1;
        let limit = end - 1;
        let cap = aztec_code_size * aztec_code_size - 11 * 11;
        let mut dominos = Vec::with_capacity(cap);

        for row in 0..limit {
            dominos.push(Domino::right((row, 0)))
        }

        for col in 0..limit {
            dominos.push(Domino::up((end, col)))
        }

        for row in 0..limit {
            dominos.push(Domino::left((end - row, end)))
        }

        for col in 0..limit {
            dominos.push(Domino::down((0, end - col)))
        }

        AztecWriter { size: aztec_code_size, dominos, current_domino: 0 }
    }
    
    pub fn into_aztec(self) -> AztecCode {
        let mut code = AztecCode::new(self.size);

        for domino in &self.dominos {
            println!("{:?}", domino);
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
        println!("Writing {} (remaining: {})", nb_dominos, remaining);

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
