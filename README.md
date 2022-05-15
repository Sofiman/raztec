# Raztec
#### **Still work in progress**

Aztec barcode reader and writer written in Rust.
The objective is that no third-party library should be used and code generation
should be fast and accurate.

## Quick start

The Aztec code generator comes as a Builder:
```Rust
use raztec::writer::AztecCodeBuilder;
let code = AztecCodeBuilder::new().error_correction(23)
    .append("Hello").append(", ").append("World!").build().unwrap();
```
Please note that the `build` function returns a **Result** as the Aztec Code
generation may fail.

This gives you an AztecCode struct that have the Index IndexMut and Display
traits. You can get the current side size with `code.size()`.
You can convert the AztecCode struct into a pixel array using `to_image`,
`to_rgb8` and `to_mono8`.

There is currently no builtin Aztec code reader.

### Generated using raztec

![Example Aztec Code](https://i.imgur.com/HmgLg70.png)

## TODO

- Implement an Aztec code reader
- Add more documentation
- Improve speed of big Aztec Code generation
