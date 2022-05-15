#[derive(Debug, PartialEq)]
pub enum AztecReadError {
    NotFound,
}

pub struct Marker {
    pub color: u32,
    pub loc: (usize, usize),
    pub size: (usize, usize)
}

impl Marker {
    pub fn red((row, col): (usize, usize), (width, height): (usize, usize)) -> Self {
        Marker { color: 0xff0000, loc: (row, col), size: (width, height) }
    }

    pub fn orange((row, col): (usize, usize), (width, height): (usize, usize)) -> Self {
        Marker { color: 0xffb86c, loc: (row, col), size: (width, height) }
    }
}

pub struct AztecReader {
    width: usize,
    height: usize,
    image: Vec<bool>,
    pub markers: Vec<Marker>
}

fn grayscale(rgb: &u32) -> u8 {
    let   red = (rgb & 0xFF0000) as f32;
    let green = (rgb & 0x00FF00) as f32;
    let  blue = (rgb & 0x0000FF) as f32;
    ((0.2126 * red + 0.7152 * green + 0.0722 * blue) * 255.0).round() as u8
}

impl AztecReader {
    pub fn from_rgb((width, height): (u32, u32), rgb: &[u32]) -> AztecReader {
        AztecReader { 
            width: width as usize, height: height as usize,
            image: rgb.iter().map(grayscale).map(|x| x < 104).collect(),
            markers: vec![]
        }
    }

    pub fn from_grayscale((width, height): (u32, u32), rgb: &[u8]) -> AztecReader {
        AztecReader { 
            width: width as usize, height: height as usize, 
            image: rgb.iter().map(|&x| x < 104).collect(),
            markers: vec![]
        }
    }

    fn check_ratio(&self, counts: &[usize; 9]) -> bool {
        let mut total = 0;
        for &count in counts {
            if count == 0 {
                return false;
            }
            total += count;
        }

        if total < 11 {
            return false;
        }

        let mod_size = (total / 11 + if total % 11 != 0 { 1 } else { 0 }) as i32;
        let max_v = mod_size as i32 / 2;

        total = 0;
        let mut i = 0;
        while i < 9 {
            if (mod_size - counts[i] as i32).abs() < max_v {
                total += 1;
            }
            i += 1;
        }
        total >= 8
    }

    fn find_bullseye(&mut self) -> Result<(usize, usize), AztecReadError> {
        let mut counts = [0; 9];
        for row in 0..self.height {
            counts.fill(0);
            let mut current_state = 0;

            for col in 0..self.width {
                if self.image[row * self.width + col] { // black pixel
                    if current_state % 2 == 1 {
                        current_state += 1;
                    }
                    counts[current_state] += 1;
                } else if current_state % 2 == 1 {
                    counts[current_state] += 1;
                } else if current_state == 8 {
                    if self.check_ratio(&counts) {
                        let w = counts.iter().sum();
                        self.markers.push(Marker::red((row, col - w), (w, 1)));
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
        Result::Err(AztecReadError::NotFound)
    }

    pub fn read(&mut self) -> Result<String, AztecReadError> {
        let (center_x, center_y) = self.find_bullseye()?;
        println!("center: (x: {}, y: {})", center_x, center_y);
        todo!();
    }
}

