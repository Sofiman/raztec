use image::ImageBuffer;
use std::io::{Write, stdout, stdin};
use std::time::Instant;

use raztec::writer::build_rune;

fn main() {
    // get input from the console (text to encode)
    let mut input = String::new();
    let val;
    loop {
        print!("Enter the Aztec Rune value (0-255): ");
        stdout().flush().unwrap();
        stdin().read_line(&mut input).unwrap();
        if let Ok(x) = input.trim().parse::<u8>() {
            val = x;
            break
        }
        input.clear();
    };

    let start = Instant::now();
    // generate the Aztec Code
    let code = build_rune(val);
    println!("Successfully generated Aztec Rune in {:?}", start.elapsed());

    // save the Aztec code as an image
    let pixels = code.to_mono8(4);
    let size = code.size() * 4;
    let img = ImageBuffer::from_fn(size as u32, size as u32, |x, y| {
        image::Luma([pixels[y as usize * size + x as usize]])
    });
    img.save("rune.png").unwrap();
    println!("Saved to rune.png");
}
