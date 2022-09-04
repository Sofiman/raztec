use show_image::{ImageView, ImageInfo, create_window, event};
use std::time::Instant;
use std::env;

use raztec::reader::{AztecCodeDetector, filters};

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let img = image::open(args.get(1).unwrap()).unwrap().into_luma8();
    let pixels: Vec<u8> = img.enumerate_pixels().map(|(_, _, p)| p[0]).collect();
    let (w, h) = img.dimensions();

    println!("Finding Aztec Codes...");
    let start = Instant::now();
    let bw = filters::process_image(&pixels);
    let mut finder = AztecCodeDetector::raw((w, h), bw.clone());
    let candidates = finder.detect_codes();
    println!("Found {} candidates in {:?}", candidates.len(), start.elapsed());
    println!("Here is the results:");
    let mut markers = vec![];
    for candidate in candidates {
        println!("[Possible code at {}]", candidate.as_marker());
        markers.push(candidate.as_marker());
        let start = Instant::now();
        let code = finder.decode(candidate);
        println!("Decoded code in {:?}", start.elapsed());
        match code {
            Ok(code) => {
                println!("=> Valid {:?} Aztec code at {:?} (size: {})",
                    code.code_type(), code.location(), code.size());
                println!("=> Result: `{}`", std::str::from_utf8(code.data())
                    .unwrap_or(&format!("<raw>{:?}", code.data())));
                if !code.features().is_empty() {
                    println!("=> Features: {:?}", code.features());
                }
            },
            Err(kind) => println!("=> Invalid code, {}", kind)
        }
    }

    let mut pixels: Vec<u32> = bw.iter()
        .map(|&x| if x { 0xFFFFFF } else { 0 }).collect();
    for marker in finder.markers().iter().chain(markers.iter()) {
        let (col, row) = marker.loc();
        let (mw, mh) = marker.size();
        for i in row..(row+mh) {
            for j in col..(col+mw) {
                pixels[i * w as usize + j] = marker.color();
            }
        }
    }

    let pixels: Vec<u8> = pixels.into_iter()
        .flat_map(|x| [(x >> 16) as u8, (x >> 8) as u8, x as u8]).collect();
    let marked = ImageView::new(ImageInfo::rgb8(w, h), &pixels);

    // Create a window with default options and display the image.
    let window = create_window("image", Default::default())?;
    window.set_image("result", marked)?;

    for event in window.event_channel()? {
      if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
                break;
            }
        }
    }

    Ok(())
}
