use image::ImageBuffer;
use std::time::Instant;

use raztec::writer::AztecCodeBuilder;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)
        .unwrap();
    let start = Instant::now();
    let code = AztecCodeBuilder::new()
        .append(input.trim()).build();
    println!("Successfully generated Aztec Code in {:?}", start.elapsed());
    let pixels = code.to_rgb(4);
    let size = code.size() * 4;
    let img = ImageBuffer::from_fn(size as u32, size as u32, |x, y| {
        image::Luma([pixels[y as usize * size + x as usize]])
    });
    img.save("example.png").unwrap();
    println!("Saved to example.png");
}
