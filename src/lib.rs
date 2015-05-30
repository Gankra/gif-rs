//! THE GIF STANDARD:
//!
//! See http://giflib.sourceforge.net/whatsinagif/ for detailed docs.
//!
//! Gif is full of tons of junk that we don't really care about, so here's a rough sketch of the
//! spec and features we care about:
//!
//! **GIF is a Little Endian format**
//!
//! For brevity and easy search:
//! * GCT = Global Colour Table
//! * LCT = Local Colour Table
//!
//! ## Image Prelude (only once, at the start of the file)
//!
//! * Header - 6 bytes
//!     * "GIF87a" or "GIF89a"
//! * Logical Screen Descriptor - 7 bytes
//!     * bytes 0-3: LEGACY GARBAGE about canvas dimensions
//!     * byte 4:    GCT MEGA FIELD
//!         * bit 0:    GCT flag (whether there will be one)
//!         * bits 1-3: GCT resolution --  ??????
//!         * bit 4:    LEGACY GARBAGE about sorting
//!         * bits 5-7: GCT size -- k -> 2^(k + 1) entries in the GCT
//!     * byte 5:    GCT background colour index
//!     * byte 6:    LEGACY GARBAGE about non-square pixels
//! * If the GCT flag was set, the GCT follows (otherwise just go to the next section)
//!     * Array of RGB triples (1 byte per channel = 3 bytes per colour)
//!     * GCT size = k -> 3*2^(k + 1) bytes
//!
//!
//! ## Image body (this loops forever)
//!
//!     * First byte: what's the next thing?
//!         * '3B': End of file -- We're done!!!!
//!         * '21': An Extension, type determined by the next byte
//!             * '01': Plain Text Extension - variable length LEGACY GARBAGE
//!                 * read the byte
//!                     * if it's 0 we're done
//!                     * otherwise cur += data[cur] + 1; loop
//!             * 'FE': Comment Extension - variable length LEGACY GARBAGE
//!                 * read the byte
//!                     * if it's 0 we're done
//!                     * otherwise cur += data[cur] + 1; loop
//!             * 'FF': Application Extension (looping!) - 17 bytes
//!                 * byte 0:      'OB'
//!                 * bytes 1-11:  "NETSCAPE2.0" (yes, really!)
//!                 * byte 12:     '03'
//!                 * byte 13:     '01'
//!                 * bytes 14-15: u16 number of times to play, 0 = "forever"
//!                 * byte 16:     '00'
//!             * 'F9': Graphics Control Extension (transparency and frame length!) - 6 bytes
//!                 * byte 0:    '04'
//!                 * byte 1:    RENDERING MEGA FIELD
//!                     * bits 0-2: reserved LEGACY GARBAGE
//!                     * bits 3-5: disposal method
//!                         * 0: unspecified (this image is not animated?)
//!                         * 1: draw this frame on top of the current image
//!                         * 2: clear image to background colour
//!                         * 3: draw this frame on top of the previous image (does this happen?)
//!                     * bit 6:    LEGACY GARBAGE about user input
//!                     * bit 7:    transparent colour flag
//!                 * bytes 2-3: u16 frame length in hundredths-of-a-second
//!                 * byte 4:    transparent colour index
//!                     * if transparent flag set, interpret this colour index as 100% transparent
//!                 * byte 5: '00'
//!         * '2C': An Actual Image:
//!             * image descriptor info - 9 bytes
//!                 * bytes 0-7: (x: u16, y: u16, width: u16, height: u16)
//!                 * byte 8: LCT MEGA FIELD
//!                     * bit 0:    LCT flag (whether there will be one)
//!                     * bit 1:    Interlace flag (whether the image data is interlaced)
//!                     * bits 2-4: LEGACY GARBAGE about sorting/reserved
//!                     * bits 5-7: LCT size -- k -> 2^(k + 1) entries in the LCT
//!             * If the LCT flag was set, the LCT follows (otherwise just go to the next section)
//!                 * Same format as GCT; see above
//!             * Image Data: this is a variable-length pseudo-LZW encoding
//!                 * byte 0:   minimum LZW code size (used in decompression)
//!                 * blocks of LZW data until you hit one that has size 0
//!                     * byte 0: block size in bytes (not including this one!)
//!                     * bytes 1-n: LZW'd data (see below for decoding)
//!
//!
//! ## Decoding Image Data
//!
//! Image data is just a series of one-bytes indices into the local or global colour table
//! (use global if local flag wasn't set -- they can't both not be set). The indices are
//! the pixels in "english reading order": left to right, then top to bottom.
//!
//! Unfortunately for us, this data is encoded in a pseudo-LZW-compressed format. How to walk
//! through that format is explained above. Here's how to decode each block.
//!
//! TODO: Read the giflib docs and figure it out
//!
//! Recall that we acquired the *minimum code size* from parsing the image data. This is like
//! the LCT and GCT sizes. Code table has real size 2^k + 2. First 2^k are exactly colour indices,
//! last 2 are "clear" and "end of information", respectively.

// A very minimal GIF
// Header:      47 49 46 38 39 61                           - "GIF89a"
// LSD:         0A 00 0A 00   91   00   00                  - 91 = 1 001 0 001 - GCT w/ 4 colours
// GCT:         FF FF FF   FF 00 00   00 00 FF   00 00 00   - white, red, blue, black
// GraphExt:    21 F9   04   00   00 00   00   00           - no animation or transparency
// Image:       2C
//      desc:   00 00   00 00   0A 00   0A 00   00          - 10x10 img, no LCT, no interlace
//      data:   02   16   8C 2D 99 87 2A 1C DC 33 A0 02 75 EC 95 FA A8 DE 60 8C 04 91 4C 01   00
//      data block (thing that starts with 8C) in binary:
//      10001100001011011001100110000111001010100001110
//      01101110000110011101000000000001001110101111011
//      00100101011111101010101000110111100110000010001
//      10000000100100100010100110000000001
//
//
//
//
//
// EOF:         3B

const HEADER_LEN: usize = 6;
const GLOBAL_DESCRIPTOR_LEN: usize = 7;
const LOCAL_DESCRIPTOR_LEN: usize = 9;
const GRAPHICS_EXTENSION_LEN: usize = 6;
const APPLICATION_EXTENSION_LEN: usize = 17;

const BLOCK_EOF: u8       = 0x3B;
const BLOCK_EXTENSION: u8 = 0x21;
const BLOCK_IMAGE: u8     = 0x2C;

const EXTENSION_PLAIN: u8       = 0x01;
const EXTENSION_COMMENT: u8     = 0xFE;
const EXTENSION_GRAPHICS: u8    = 0xF9;
const EXTENSION_APPLICATION: u8 = 0xFF;

pub struct Gif {
    pub width: u16,
    pub height: u16,
    pub frames: Vec<Frame>,
}

pub struct Frame {
    pub width: u16,
    pub height: u16,
    pub data: Vec<u8>
}


pub fn parse_gif(data: &[u8]) -> Result<Gif, &'static str> {
    // Note: this code frequently creates new "data" variables to represent all the bytes we
    // haven't read yet. This is just more convenient than re-assigning `data` or tracking our
    // position otherwise.

    // ~~~~~~~~~ Image Prelude ~~~~~~~~~~

    if data.len() < HEADER_LEN + GLOBAL_DESCRIPTOR_LEN { return Err("File too small to be a GIF"); }

    let (header, data) = data.split_at(HEADER_LEN);
    if header != b"GIF87a" && header != b"GIF89a" { return Err("Not a GIF"); }

    let (descriptor, data) = data.split_at(GLOBAL_DESCRIPTOR_LEN);
    let gct_mega_field = descriptor[4];
    let gct_background_color_index = descriptor[5];
    let gct_flag = (gct_mega_field & 0b1000_0000) != 0;
    let gct_size_exponent = gct_mega_field & 0b0000_0111;
    let gct_size = 1usize << (gct_size_exponent + 1); // 2^(k+1)

    let (gct, data) = if gct_flag {
        if data.len() < 3 * gct_size { return Err("Unexpected end of GIF"); }
        let (gct, data) = data.split_at(3 * gct_size);
        (Some(gct), data)
    } else {
        (None, data)
    };

    // ~~~~~~~~~ Image Body ~~~~~~~~~~~

    let mut temp_data = data;
    let mut transparent_index = None;
    let mut num_iterations = None;
    let mut frame_delay = 0;
    let mut disposal_method = 0;
    let mut frames = vec![];
    let mut width = 0;
    let mut height = 0;

    loop {
        let data = temp_data;
        if data.len() < 1 { return Err("Unexpected end of GIF"); }
        match data[0] {
            BLOCK_EOF => {
                // TODO: check if this was a sane place to stop?
                return Ok(Gif {
                    width: width as u16,
                    height: height as u16,
                    frames: frames,
                })
            }
            BLOCK_EXTENSION => {
                // 3 to coalesce some checks we'll have to make in any branch
                if data.len() < 3 { return Err("Unexpected end of GIF"); }
                match data[1] {
                    EXTENSION_PLAIN | EXTENSION_COMMENT => {
                        // This is legacy garbage, but has a variable length so
                        // we need to parse it a bit to get over it.
                        let mut data = &data[2..];
                        loop {
                            let to_skip = data[0] as usize;
                            if to_skip == 0 { break; }
                            if data.len() < to_skip + 2 { return Err("Unexpected end of GIF"); }
                            data = &data[to_skip+1..];
                        }
                        temp_data = &data[1..];
                    }
                    EXTENSION_GRAPHICS => {
                        // Frame delay and transparency settings
                        let data = &data[2..];
                        if data.len() < GRAPHICS_EXTENSION_LEN {
                            return Err("Unexpected end of GIF");
                        }
                        let (ext, data) = data.split_at(GRAPHICS_EXTENSION_LEN);

                        let rendering_mega_field = ext[1];
                        let transparency_flag = (rendering_mega_field & 0b0000_0001) != 0;

                        disposal_method = (rendering_mega_field & 0b0001_1100) >> 2;
                        frame_delay = le_u16(ext[2], ext[3]);
                        transparent_index = if transparency_flag {
                            Some(ext[4])
                        } else {
                            None
                        };

                        temp_data = data;
                    }
                    EXTENSION_APPLICATION => {
                        // NETSCAPE 2.0 Looping Extension

                        let data = &data[2..];
                        if data.len() < APPLICATION_EXTENSION_LEN {
                            return Err("Unexpected end of GIF");
                        }

                        // TODO: verify this is the desired application extension?
                        let (ext, data) = data.split_at(APPLICATION_EXTENSION_LEN);

                        num_iterations = Some(le_u16(ext[14], ext[15]));

                        temp_data = data;
                    }
                    _ => { return Err("Unknown extension type found"); }
                }
            }
            BLOCK_IMAGE => {
                let data = &data[1..];
                if data.len() < LOCAL_DESCRIPTOR_LEN { return Err("Unexpected end of GIF"); }
                let (descriptor, data) = data.split_at(LOCAL_DESCRIPTOR_LEN);
                println!("local image desc {:?}", descriptor);
                let x      = le_u16(descriptor[0], descriptor[1]) as usize;
                let y      = le_u16(descriptor[2], descriptor[3]) as usize;
                width  = le_u16(descriptor[4], descriptor[5]) as usize;
                height = le_u16(descriptor[6], descriptor[7]) as usize;

                let lct_mega_field = descriptor[8];
                let lct_flag = (lct_mega_field & 0b1000_0000) != 0;
                let interlace = (lct_mega_field & 0b0100_0000) != 0;
                let lct_size_exponent = (lct_mega_field & 0b0000_1110) >> 1;
                let lct_size = 1usize << (lct_size_exponent + 1); // 2^(k+1)

                let (lct, data) = if lct_flag {
                    if data.len() < 3 * lct_size { return Err("Unexpected end of GIF"); }
                    let (lct, data) = data.split_at(3 * lct_size);
                    (Some(lct), data)
                } else {
                    (None, data)
                };

                if data.len() < 2 { return Err("Unexpected end of GIF"); }
                let minimum_code_size = data[0];
                let mut data = &data[1..];

                let mut indices = vec![0; width * height]; //TODO: not this

                let mut parse_state = create_parse_state(minimum_code_size, width * height);


                // ~~~~~~~~~~~~~~ DECODE THE INDICES ~~~~~~~~~~~~~~~~

                if interlace {
                    let interlaced_offset = [0, 4, 2, 1];
                    let interlaced_jumps = [8, 8, 4, 2];
                    for i in 0..4 {
                        let mut j = interlaced_offset[i];
                        while j < height {
                            let offset = j * width; // * 4?
                            get_line(&mut parse_state, &mut indices[j * width..], width, &mut data);
                            j += interlaced_jumps[i];
                        }
                    }
                } else {
                    get_line(&mut parse_state, &mut indices, width * height, &mut data);
                }

                // ~~~~~~~~~~~ MAP THE INDICES TO COLOURS ~~~~~~~~~~~

                let mut pixels = vec![0; width * height * 4]; // RGBA

                let color_map = lct.as_ref().unwrap_or_else(|| gct.as_ref().unwrap());
                for (pix_idx, col_idx) in indices.into_iter().enumerate() {
                    let is_transparent = transparent_index.map(|idx| idx == col_idx)
                                                          .unwrap_or(false);
                    let (r, g, b, a) = if is_transparent {
                        (0, 0, 0, 0)
                    } else {
                        let col_idx = col_idx as usize;
                        let r = color_map[col_idx * 3 + 0];
                        let g = color_map[col_idx * 3 + 1];
                        let b = color_map[col_idx * 3 + 2];
                        (r, g, b, 0xFF)
                    };
                    pixels[pix_idx * 4 + 0] = r;
                    pixels[pix_idx * 4 + 1] = g;
                    pixels[pix_idx * 4 + 2] = b;
                    pixels[pix_idx * 4 + 3] = a;
                }
                // TODO: use the processed data (disposal method, etc)

                frames.push(Frame {
                    width: width as u16,
                    height: height as u16,
                    data: pixels
                });

                // TODO: reset the fields
                temp_data = &data[1..];
                transparent_index = None;
                frame_delay = 0;
                disposal_method = 0;
            }
            x => {
                println!("Bad block: {:02X}", x);
                return Err("Unknown block type found");
            }
        }

    }
}


const LZ_MAX_CODE: usize = 4095;
const LZ_BITS: usize = 12;

const NO_SUCH_CODE: usize = 4098;    /* Impossible code, to signal empty. */

struct ParseState {
    bits_per_pixel: usize,
    clear_code: usize,
    eof_code: usize,
    running_code: usize,
    running_bits: usize,
    max_code_1: usize,
    last_code: usize,
    stack_ptr: usize,
    current_shift_state: usize,
    current_shift_dword: usize,
    pixel_count: usize,
    buf: [u8; 256], // [0] = len, [1] = cur_index
    stack: [u8; LZ_MAX_CODE],
    suffix: [u8; LZ_MAX_CODE + 1],
    prefix: [usize; LZ_MAX_CODE + 1],
}

fn create_parse_state(code_size: u8, pixel_count: usize) -> ParseState {
    let bits_per_pixel = code_size as usize;
    let clear_code = 1 << bits_per_pixel;

    ParseState {
        buf: [0; 256], // giflib only inits the first byte to 0
        bits_per_pixel: bits_per_pixel,
        clear_code: clear_code,
        eof_code: clear_code + 1,
        running_code: clear_code + 2,
        running_bits: bits_per_pixel + 1,
        max_code_1: 1 << (bits_per_pixel + 1),
        stack_ptr: 0,
        last_code: NO_SUCH_CODE,
        current_shift_state: 0,
        current_shift_dword: 0,
        prefix: [NO_SUCH_CODE; LZ_MAX_CODE + 1],

        // TODO: verify these initializations
        suffix: [0; LZ_MAX_CODE + 1],
        stack: [0; LZ_MAX_CODE],
        pixel_count: pixel_count,
    }
}

fn get_line(state: &mut ParseState, line: &mut[u8], line_len: usize, data: &mut &[u8]) {
    state.pixel_count -= line_len;
    if state.pixel_count > 0xffff0000 {
        panic!("Gif has too much pixel data");
    }

    decompress_line(state, line, line_len, data);

    if state.pixel_count == 0 {
        // TODO: giflib seems to think there might be more gif left after
        // this is done. Should probably skip through the rest of the blocks
        // here so that the stream is in the right place.
        let bytes_left = state.buf[0] as usize;
        *data = &data[bytes_left..];

        loop {
            let to_skip = data[0] as usize;
            if to_skip == 0 { break; }
            if data.len() < to_skip + 2 { panic!("Unexpected end of GIF"); }
            *data = &data[to_skip+1..];
        }
    }
}

fn decompress_line(state: &mut ParseState, line: &mut[u8], line_len: usize, data: &mut &[u8]) {
    let mut i = 0;
    let mut current_prefix; // This is uninit in dgif
    let &mut ParseState {
        mut stack_ptr,
        eof_code,
        clear_code,
        mut last_code,
        ..
    } = state;

    assert!(stack_ptr <= LZ_MAX_CODE, "TODO: I don't know why yet");
    while stack_ptr != 0 && i < line_len {
        stack_ptr -= 1;
        line[i] = state.stack[stack_ptr];
        i += 1;
    }

    while i < line_len {
        let current_code = decompress_input(state, data);

        let &mut ParseState {
            ref mut prefix,
            ref mut suffix,
            ref mut stack,
            ..
        } = state;

        assert!(current_code != eof_code, "Unexpected end of image");

        if current_code == clear_code {
            // Reset all the sweet codez we learned
            for j in 0..LZ_MAX_CODE {
                prefix[j] = NO_SUCH_CODE;
            }

            state.running_code = state.eof_code + 1;
            state.running_bits = state.bits_per_pixel + 1;
            state.max_code_1 = 1 << state.running_bits;
            state.last_code = NO_SUCH_CODE;
            last_code = state.last_code;
        } else {
            // Regular code
            if current_code < clear_code {
                // single index code, direct mapping to a colour index
                line[i] = current_code as u8;
                i += 1;
            } else {
                // MULTI-CODE MULTI-CODE ENGAGE -- DASH DASH DASH!!!!
                if prefix[current_code] == NO_SUCH_CODE {
                    current_prefix = last_code;

                    // VIA DGIF:
                    /* Only allowed if CrntCode is exactly the running code:
                     * In that case CrntCode = XXXCode, CrntCode or the
                     * prefix code is last code and the suffix char is
                     * exactly the prefix of last code! */
                    let code = if current_code == state.running_code - 2 {
                        last_code
                    } else {
                        current_code
                    };

                    let prefix_char = get_prefix_char(&*prefix, code, clear_code) as u8;
                    stack[stack_ptr] = prefix_char;
                    suffix[state.running_code - 2] = prefix_char;
                    stack_ptr += 1;
                } else {
                    current_prefix = current_code;
                }

                /* Now (if image is O.K.) we should not get a NO_SUCH_CODE
                 * during the trace. As we might loop forever, in case of
                 * defective image, we use StackPtr as loop counter and stop
                 * before overflowing Stack[]. */

                while stack_ptr < LZ_MAX_CODE
                        && current_prefix > clear_code
                        && current_prefix <= LZ_MAX_CODE {

                    stack[stack_ptr] = suffix[current_prefix];
                    stack_ptr += 1;
                    current_prefix = prefix[current_prefix];
                }

                if stack_ptr >= LZ_MAX_CODE || current_prefix > LZ_MAX_CODE {
                    panic!("Image defect???");
                }

                stack[stack_ptr] = current_prefix as u8;
                stack_ptr += 1;

                while stack_ptr != 0 && i < line_len {
                    stack_ptr -= 1;
                    line[i] = stack[stack_ptr];
                    i += 1;
                }

            }

            if last_code != NO_SUCH_CODE && prefix[state.running_code - 2] == NO_SUCH_CODE {
                prefix[state.running_code - 2] = last_code;

                let code = if current_code == state.running_code - 2 {
                    /* Only allowed if CrntCode is exactly the running code:
                     * In that case CrntCode = XXXCode, CrntCode or the
                     * prefix code is last code and the suffix char is
                     * exactly the prefix of last code! */
                    last_code
                } else {
                    current_code
                };

                suffix[state.running_code - 2] = get_prefix_char(&*prefix, code, clear_code) as u8;
            }

            last_code = current_code;
        }
    }

    state.last_code = last_code;
    state.stack_ptr = stack_ptr;
}

// Prefix is a virtual linked list or something.
fn get_prefix_char(prefix: &[usize], mut code: usize, clear_code: usize) -> usize {
    let mut i = 0;

    loop {
        if code <= clear_code { break; }
        i += 1;
        if i > LZ_MAX_CODE { break; }
        if code > LZ_MAX_CODE { return NO_SUCH_CODE; }
        code = prefix[code];
    }

    return code;
}

fn decompress_input(state: &mut ParseState, src: &mut &[u8]) -> usize {
    let code_masks: [usize; 13] = [
        0x0000, 0x0001, 0x0003, 0x0007,
        0x000f, 0x001f, 0x003f, 0x007f,
        0x00ff, 0x01ff, 0x03ff, 0x07ff,
        0x0fff
    ];

    assert!(state.running_bits <= LZ_BITS); // Make an Err later

    while state.current_shift_state < state.running_bits {
        // Get the next byte, which is either in this block or the next one
        let next_byte = if state.buf[0] == 0 {
            // This block is done, get the next one
            let len = src[0] as usize;
            state.buf[0] = len as u8;

            assert!(len != 0 && src.len() > len, "Unexpected end of GIF");

            // Copy next len bytes from src to buf
            for i in 0..len {
                state.buf[i + 1] = src[i + 1];
            }

            // seek ahead
            *src = &src[len + 1..];

            let next_byte = state.buf[1];
            state.buf[1] = 2;
            state.buf[0] -= 1;
            next_byte
        } else {
            // Still got bytes in this block
            let next_byte = state.buf[state.buf[1] as usize];
            state.buf[1] += 1;
            state.buf[0] -= 1;
            next_byte
        };

        state.current_shift_dword |= (next_byte as usize) << state.current_shift_state;
        state.current_shift_state += 8;
    }

    let code = state.current_shift_dword & code_masks[state.running_bits];
    state.current_shift_dword >>= state.running_bits;
    state.current_shift_state -= state.running_bits;

    if state.running_code < LZ_MAX_CODE + 2 {
        state.running_code += 1;
        if state.running_code > state.max_code_1 && state.running_bits < LZ_BITS {
            state.max_code_1 <<= 1;
            state.running_bits += 1;
        }
    }

    code
}

fn le_u16(first: u8, second: u8) -> u16 {
    // TODO: verify this; I am bad at endian stuff
    ((second as u16) << 8) | (first as u16)
}

