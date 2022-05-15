use image::ImageBuffer;
use show_image::{ImageView, ImageInfo, create_window, event};
use std::env;

use raztec::reader::AztecReader;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let img = image::open(args.get(1).unwrap()).unwrap().into_luma8();
    let mut pixels = Vec::new();
    for (_x, _y, p) in img.enumerate_pixels() {
        pixels.push(p[0]);
    }
    let mut reader = AztecReader::from_grayscale(img.dimensions(), &pixels);
    reader.read();

    let (width, height) = img.dimensions();
    let pixels: Vec<u8> = pixels.iter().map(|&x| x <= 104)
        .map(|x| if x { 0 } else { 255 }).collect();
    let image = ImageView::new(ImageInfo::mono8(width, height), &pixels);

    let mut pixels: Vec<u32> = pixels.iter().map(|&x| x <= 51)
        .map(|x| if x { 0 } else { 0xffffff }).collect();
    for marker in reader.markers.iter() {
        let (row, col) = marker.loc;
        let (w, h) = marker.size;
        for i in row..(row+h) {
            for j in col..(col+w) {
                pixels[i * width as usize + j] = (pixels[i * width as usize + j] + marker.color) / 2;
            }
        }
    }

    let pixels: Vec<u8> = pixels.into_iter()
        .flat_map(|x| [(x >> 16) as u8, (x >> 8) as u8, x as u8]).collect();
    let marked = ImageView::new(ImageInfo::rgb8(width, height), &pixels);

    // Create a window with default options and display the image.
    let window = create_window("image", Default::default())?;
    //window.set_image("image-001", image)?;
    window.set_image("marked", marked)?;

    for event in window.event_channel()? {
      if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
                break;
            }
        }
    }

    Ok(())
}
