//! Image processing helper functions

/// Contrast Stretching
pub fn normalize_image(buffer: &mut [u8]) -> (u8, u8) {
    let mut mi = 255;
    let mut ma = 0;
    for &v in buffer.iter() {
        if v > ma {
            ma = mi;
        }
        if v < mi {
            mi = v;
        }
    }
    let extent = (ma - mi) as u32;
    for pix in buffer {
        *pix = (((*pix - mi) as u32 * 255) / extent) as u8;
    }
    (mi, ma)
}

/// Linear Transform => for each pixel: alpha * (pixel) + beta
/// where **alpha** is the constrast control
///   and **beta** is the brightness control
pub fn linear_transform(buffer: &mut[u8], alpha: f32, beta: f32) {
    buffer.iter_mut().for_each(|pix| {
        *pix = ((alpha * (*pix as f32) + beta).round().min(255.0)) as u8;
    });
}

/// Black and White Linear Transform => for each pixel: alpha * (pixel) + beta
/// where **alpha** is the constrast control
///       **beta** is the brightness control
///   and **threshold** where any pixel value over will be considered as white
pub fn bw_transform(buffer: &[u8], alpha: f32, beta: f32, threshold: f32)
    -> Vec<bool> {
    buffer.iter().map(|&pix| alpha * (pix as f32) + beta < threshold).collect()
}

/// Auto contrast algorithm. See <https://stackoverflow.com/a/9761841>.
/// Returns (alpha, beta) for a linear transform
pub fn auto_contrast(buffer: &[u8]) -> (f32, f32) {
    let mut histogram = [0usize; 256];
    // build intensity histogram
    for &v in buffer.iter() {
        histogram[v as usize] += 1;
    }

    // find 5th and 95th percentile
    let target5 = buffer.len() * 5 / 100;
    let target95 = buffer.len() - target5;
    let mut p95 = 0;
    let mut p5 = 0;
    let mut sum: usize = 0;

    for (i, &val) in histogram.iter().enumerate() {
        sum += val;
        if p5 == 0 {
            if sum > target5 {
                p5 = i;
            }
        } else if p95 == 0 {
            if sum > target95 {
                p95 = i;
            }
        } else {
            break;
        }
    }
    if p5 == 0 && p95 == 255 { // no enhancement possible
        return (1.0, 0.0);
    }
    if p95 == 0 { // unreachable (or black image)
        return (1.0, 0.0);
    }

    // apply a linear transform to stretch pixel values
    let p95 = p95 as f32;
    let ratio = p5 as f32 / p95;
    let beta = -255.0 * ratio / (1.0 - ratio);
    let alpha = (255.0 - beta) / p95;
    (alpha, beta)
}

/// Convert a grayscale image into a black and white image using the previously
/// defined auto-contrast function
pub fn process_image(mono: &[u8]) -> Vec<bool> {
    let (alpha, beta) = auto_contrast(mono);
    bw_transform(mono, alpha, beta, 104.0)
}
