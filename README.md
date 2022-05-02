# Raztec
#### **Still work in progress**

Aztec barcode reader and writer written in Rust.
The objective is that no third-party library should be used and code generation
should be fast and accurate.

## Quick start

The Aztec code generator comes as a Builder:
```Rust
use raztec::writer::AztecCodeBuilder;
let code = AztecCodeBuilder::new(23)
        .append("Hello").append(", ").append("World!").build(); // AztecCode
```
This gives you an AztecCode struct that have the Index IndexMut and Display
traits. You can get the current side size with `code.size()`.

There is currently no builtin Aztec code reader.

### Generated using raztec

```
██████████████████████████████████████████
██  ████████          ██  ████████████████
██  ██  ██      ██  ██████  ████    ██  ██
████      ██    ████  ██  ████          ██
██    ██        ██  ██      ████████  ████
████████      ██  ████      ██  ████  ████
████████                        ████    ██
████  ████    ██████████████            ██
████    ████  ██          ██  ██      ████
██  ██████    ██  ██████  ██  ██        ██
██  ██  ████  ██  ██  ██  ██    ██  ██████
████████████  ██  ██████  ██  ████  ██████
████████      ██          ██    ██    ████
██  ████      ██████████████    ██████████
██  ██    ██                        ██████
████    ██████    ██  ██████████      ████
████      ████  ██  ██        ██    ██████
████  ████████  ██      ██████  ██      ██
████████████████  ██  ██████        ██████
████████  ██  ████  ██  ██    ██  ██  ████
██████████████████████████████████████████
```
