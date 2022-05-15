use image::ImageBuffer;
use std::time::Instant;

use raztec::writer::AztecCodeBuilder;

fn main() {
    // get input from the console (text to encode)
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)
        .unwrap();

    let start = Instant::now();
    // generate the Aztec Code
    let code = AztecCodeBuilder::new()
        .append(input.trim()).build().unwrap();
    println!("Successfully generated Aztec Code in {:?}", start.elapsed());

    // save the Aztec code as an image
    let pixels = code.to_mono8(4);
    let size = code.size() * 4;
    let img = ImageBuffer::from_fn(size as u32, size as u32, |x, y| {
        image::Luma([pixels[y as usize * size + x as usize]])
    });
    img.save("example.png").unwrap();
    println!("Saved to example.png");
}
