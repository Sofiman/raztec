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

This library also supports the generation and scanning of Aztec Runes.
An Aztec Rune is a compact Aztec Code containing only one byte of information.
Here is an example of generating a rune with the byte value 38:
```Rust
use raztec::writer::AztecCodeBuilder;
let rune = AztecCodeBuilder::build_rune(38);
```

There is currently no builtin Aztec code reader (Coming soon).

### Generated using raztec

![Example Aztec Code](https://i.imgur.com/HmgLg70.png)

## Documentation

The library's code is fully documented, to see the documentation use:
```cargo doc --open```

## Issues

If you enconter any issues or bugs while generating or scanning Aztec Codes,
please open an issue so I can look into it.

## TODO

- Implement an Aztec code reader
- Improve speed of big Aztec Code generation
