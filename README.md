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
    .append("Hello").append(", ").append("World!").build();
```
This gives you an AztecCode struct that have the Index IndexMut and Display
traits. You can get the current side size with `code.size()`.

There is currently no builtin Aztec code reader.

### Generated using raztec

![Example Aztec Code](https://i.imgur.com/HmgLg70.png)

## TODO

- Add support for larger Aztec Codes (full-size)
- Add support for ECI characters
- Implement an Aztec code reader
