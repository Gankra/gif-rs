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
//!     * bytes 0-3: (width: u16, height: u16)
//!     * byte 4:    GCT MEGA FIELD
//!         * bit 0:    GCT flag (whether there will be one)
//!         * bits 1-3: GCT resolution -- ??????
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
//!                     * note that this means "reuse the old pixel" if using disposal current/prev
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
//!                     * bytes 1-n: LZW'd data
//!
//!
//! ## Decoding Image Data
//!
//! Image data is just a series of one-bytes indices into the local or global colour table
//! (use global if local flag wasn't set -- they can't both not be set). The indices are
//! the pixels in "english reading order": left to right, then top to bottom.
//!
//! Unfortunately for us, this data is encoded in a pseudo-LZW-compressed format. How to walk
//! through that format is explained above. Decoding it is super complicated though, and I really
//! don't understand it. This is the one place I basically deferred to GIFLIB's implementation
//! with occassional cleanups.
//!
//! A very minimal GIF
//! Header:      47 49 46 38 39 61                           - "GIF89a"
//! LSD:         0A 00 0A 00   91   00   00                  - 91 = 1 001 0 001 - GCT w/ 4 colours
//! GCT:         FF FF FF   FF 00 00   00 00 FF   00 00 00   - white, red, blue, black
//! GraphExt:    21 F9   04   00   00 00   00   00           - no animation or transparency
//! Image:       2C
//!      desc:   00 00   00 00   0A 00   0A 00   00          - 10x10 img, no LCT, no interlace
//!      data:   02   16   8C 2D 99 87 2A 1C DC 33 A0 02 75 EC 95 FA A8 DE 60 8C 04 91 4C 01   00
//!      data block (thing that starts with 8C) in binary:
//! EOF:         3B

use std::io::{Read, Seek, SeekFrom};
use std::io::Error as IoError;

const HEADER_LEN: usize = 6;
const GLOBAL_DESCRIPTOR_LEN: usize = 7;
const LOCAL_DESCRIPTOR_LEN: usize = 9;
const GRAPHICS_EXTENSION_LEN: usize = 6;
const APPLICATION_EXTENSION_LEN: usize = 17;
const MAX_COLOR_TABLE_SIZE: usize = 3 * 256; // 2^8 RGB colours

const BLOCK_EOF: u8       = 0x3B;
const BLOCK_EXTENSION: u8 = 0x21;
const BLOCK_IMAGE: u8     = 0x2C;

const EXTENSION_PLAIN: u8       = 0x01;
const EXTENSION_COMMENT: u8     = 0xFE;
const EXTENSION_GRAPHICS: u8    = 0xF9;
const EXTENSION_APPLICATION: u8 = 0xFF;

const DISPOSAL_UNSPECIFIED: u8 = 0;
const DISPOSAL_CURRENT: u8 = 1;
const DISPOSAL_BG: u8 = 2;
const DISPOSAL_PREVIOUS: u8 = 3;

const BYTES_PER_COL: usize = 4;

pub struct Gif<R> {
    width: u16,
    height: u16,

    /// Defaults to 1, but at some point we may discover a new value.
    /// Presumably this should only happen once.
    num_iterations: u16,
    frames: Vec<Frame>,
    animated: bool,
    fully_parsed: bool,

    gct_bg: usize,
    gct: Option<Box<[u8; MAX_COLOR_TABLE_SIZE]>>,
    data: R,
    parse_state: Option<Box<LzwParseState>>,
    stream_pos: u64,
}

pub struct Frame {
    time: u32,
    duration: u32,
    stream_pos: u64,
    is_key: bool,
    parse_state: ParseState,

    // info we *might* find while parsing metadata
    transparent_index: Option<u8>,
    disposal_method: u8,

    x: u16,
    y: u16,
    width: u16,
    height: u16,
    interlace: bool,
    lct: Option<Box<[u8; MAX_COLOR_TABLE_SIZE]>>,

    indices: Option<Box<[u8]>>, // len = width * height
}

pub struct GifView<'a, R: 'a> {
    gif: &'a mut Gif<R>,
    current_frame: usize,
}

#[derive(Debug)]
pub enum Error {
    IoError,
    Malformed,
    EndOfFile,
}

enum ParseState {
    Fresh,
    ExtensionsParsed,
    MetadataParsed,
    Parsed,
}

impl From<IoError> for Error {
    fn from(_: IoError) -> Error { Error::IoError }
}

pub type GifResult<T> = Result<T, Error>;


impl Frame {
    fn new(time: u32) -> Frame {
        Frame {
            time: time,
            duration: 0,
            stream_pos: 0,
            is_key: false,
            parse_state: ParseState::Fresh,

            transparent_index: None,
            disposal_method: 0,

            x: 0,
            y: 0,
            width: 0,
            height: 0,
            interlace: false,
            lct: None,

            indices: None,
        }
    }

    #[inline] pub fn time(&self) -> u32 { self.time }
    #[inline] pub fn duration(&self) -> u32 { self.duration }
}

impl <'a, R: Read + Seek> GifView<'a, R> {
    /// Takes two buffers of bytes: the current frame and the previous frame.
    ///
    /// To allow the user of the library to do their own memory management, this API
    /// requires that the memory for the frames be passed in. In order to correctly
    /// render the *next* GIF frame, it is necessary that the pixels of the current
    /// and previous frame be available. The previous frame will subsequently be
    /// overwritten with the next frame's pixels.
    ///
    /// If this is the first frame to be rendered, the current_frame can be
    /// empty. Otherwise, both frames should have the size returned by `frame_size`
    /// (this value is a constant for any given Gif).
    pub fn get_next_frame(&mut self, prev_frame: &mut[u8], current_frame: &[u8])
        -> GifResult<bool>
    {
        loop {
            match self.gif.frames[self.current_frame].parse_state {
                ParseState::Fresh => {
                    let has_next_frame = try!(self.gif.parse_extension_blocks());
                    if !has_next_frame { return Ok(false); }
                }
                ParseState::ExtensionsParsed => {
                    try!(self.gif.parse_image_block_header());
                }
                ParseState::MetadataParsed => {
                    try!(self.gif.parse_indices());
                }
                ParseState::Parsed => {
                    try!(self.gif.decode_indices(prev_frame,
                                                 current_frame,
                                                 &self.gif.frames[self.current_frame]));
                    self.current_frame += 1;
                    return Ok(true);
                }
            }
        }
    }

    /// Seeks to the first keyframe before (inclusive) the given time in 100ths
    /// of a second. At this point the user of the library needs to start calling
    /// get_next_frame until they reach the desired time.
    ///
    /// WIP
    #[inline]
    pub fn seek_before(&mut self, time: u32) -> Result<u32, ()> {
        unimplemented!();
    }
}

impl<R: Read + Seek> Gif<R> {
    /// Interpret the given Reader as an entire Gif file. Parses out the
    /// prelude to get most metadata (some will show up later, maybe).
    ///
    /// Returns an Error if the stream isn't long enough to get the whole
    /// header in one shot.
    pub fn new(mut data: R) -> GifResult<Gif<R>> {

        // ~~~~~~~~~ Image Prelude ~~~~~~~~~~
        let mut buf = [0; HEADER_LEN + GLOBAL_DESCRIPTOR_LEN];
        try!(read_to_full(&mut data, &mut buf));

        let (header, descriptor) = buf.split_at(HEADER_LEN);
        if header != b"GIF87a" && header != b"GIF89a" { return Err(Error::Malformed); }

        let full_width = le_u16(descriptor[0], descriptor[1]);
        let full_height = le_u16(descriptor[2], descriptor[3]);
        let gct_mega_field = descriptor[4];
        let gct_background_color_index = descriptor[5] as usize;
        let gct_flag = (gct_mega_field & 0b1000_0000) != 0;
        let gct_size_exponent = gct_mega_field & 0b0000_0111;
        let gct_size = 1usize << (gct_size_exponent + 1); // 2^(k+1)

        let gct = if gct_flag {
            let mut gct_buf = Box::new([0; MAX_COLOR_TABLE_SIZE]);
            {
                let gct = &mut gct_buf[.. 3 * gct_size];
                try!(read_to_full(&mut data, gct));
            }
            Some(gct_buf)
        } else {
            None
        };

        let mut gif = Gif {
            width: full_width,
            height: full_height,

            // This may be changed as we parse more
            num_iterations: 1,
            animated: false,

            frames: vec![],
            gct_bg: gct_background_color_index,
            gct: gct,
            data: data,
            parse_state: None,
            fully_parsed: false,
            stream_pos: 0,
        };

        gif.save_stream_position();
        gif.frames.push(Frame::new(0));
        try!(gif.parse_extension_blocks());

        Ok(gif)
    }

    pub fn view(&mut self) -> GifView<R> {
        GifView { gif: self, current_frame: 0, }
    }

    #[inline] pub fn width(&self) -> u32 { self.width as u32 }
    #[inline] pub fn height(&self) -> u32 { self.height as u32 }
    #[inline] pub fn is_animated(&self) -> bool { self.animated }
    #[inline] pub fn num_iterations(&self) -> u16 { self.num_iterations }
    #[inline] pub fn is_transparent(&self) -> bool { true }
    #[inline] pub fn frame_size(&self) -> usize { self.width as usize * self.height as usize * 4 }

    /// Gets the duration of the whole GIF in 100ths of a second.
    ///
    /// This is a value that cannot be known until the entire GIF has been
    /// read. As such, while the gif hasn't been fully parsed, this is just
    /// a lower bound. If the gif is not animated, the value is 0.
    #[inline]
    pub fn duration(&self) -> u32 {
        self.frames.last()
                   .map(|frame| frame.time + frame.duration)
                   .unwrap_or(0)
    }

    fn last_frame(&mut self) -> &mut Frame {
        self.frames.last_mut().unwrap()
    }

    fn save_stream_position(&mut self) {
        self.stream_pos = self.data.seek(SeekFrom::Current(0)).unwrap();
    }

    fn load_stream_position(&mut self) {
        self.data.seek(SeekFrom::Start(self.stream_pos)).unwrap();
    }

    /// Reads more of the stream until an entire new frame has been computed.
    /// Returns `false` if the file ends, and `true` otherwise.
    fn parse_extension_blocks(&mut self) -> GifResult<bool> {
        self.load_stream_position();

        loop {
            match try!(read_byte(&mut self.data)) {
                BLOCK_EOF => {
                    self.frames.pop();
                    self.fully_parsed = true;
                    self.parse_state = None; // free up the memory (it's several KB)
                    return Ok(false)
                }
                BLOCK_EXTENSION => {
                    match try!(read_byte(&mut self.data)) {
                        EXTENSION_PLAIN | EXTENSION_COMMENT => {
                            // This is legacy garbage, but has a variable length so
                            // we need to parse it a bit to get over it.
                            try!(skip_blocks(&mut self.data));
                        }
                        EXTENSION_GRAPHICS => {
                            // Frame delay and transparency settings
                            let mut ext = [0; GRAPHICS_EXTENSION_LEN];
                            try!(read_to_full(&mut self.data, &mut ext));

                            let rendering_mega_field = ext[1];
                            let transparency_flag = (rendering_mega_field & 0b0000_0001) != 0;

                            self.animated = true;

                            let last_frame = self.last_frame();

                            last_frame.disposal_method =
                                (rendering_mega_field & 0b0001_1100) >> 2;

                            last_frame.duration = le_u16(ext[2], ext[3]) as u32;

                            last_frame.transparent_index = if transparency_flag {
                                Some(ext[4])
                            } else {
                                None
                            };
                        }
                        EXTENSION_APPLICATION => {
                            // NETSCAPE 2.0 Looping Extension

                            let mut ext = [0; APPLICATION_EXTENSION_LEN];
                            try!(read_to_full(&mut self.data, &mut ext));

                            // TODO: Verify this is the NETSCAPE 2.0 extension?

                            self.num_iterations = le_u16(ext[14], ext[15]);
                        }
                        _ => {
                            // unknown extension type
                            return Err(Error::Malformed);
                        }
                    }
                    self.save_stream_position();
                }
                BLOCK_IMAGE => {
                    self.last_frame().parse_state = ParseState::ExtensionsParsed;
                    self.save_stream_position();
                    return Ok(true)
                }
                _ => {
                    // unknown block type
                    return Err(Error::Malformed);
                }
            }
        }
    }

    fn parse_image_block_header(&mut self) -> GifResult<()> {
        self.load_stream_position();

        // new scope to toss frame before save_stream_position
        {
            let frame = self.frames.last_mut().unwrap();

            let mut descriptor = [0; LOCAL_DESCRIPTOR_LEN];
            try!(read_to_full(&mut self.data, &mut descriptor));


            frame.x      = le_u16(descriptor[0], descriptor[1]);
            frame.y      = le_u16(descriptor[2], descriptor[3]);
            frame.width  = le_u16(descriptor[4], descriptor[5]);
            frame.height = le_u16(descriptor[6], descriptor[7]);

            let num_indices = frame.width as usize * frame.height as usize;

            let lct_mega_field = descriptor[8];
            let lct_flag = (lct_mega_field & 0b1000_0000) != 0;
            frame.interlace = (lct_mega_field & 0b0100_0000) != 0;
            let lct_size_exponent = (lct_mega_field & 0b0000_1110) >> 1;
            let lct_size = 1usize << (lct_size_exponent + 1); // 2^(k+1)

            frame.lct = if lct_flag {
                let mut lct_buf = Box::new([0; MAX_COLOR_TABLE_SIZE]);
                {
                    let lct = &mut lct_buf[.. 3 * lct_size];
                    try!(read_to_full(&mut self.data, lct));
                }
                Some(lct_buf)
            } else {
                None
            };

            let minimum_code_size = try!(read_byte(&mut self.data));

            /* For debugging
            println!("");
            println!("starting frame decoding: {}", self.frames.len());
            println!("x: {}, y: {}, w: {}, h: {}", x, y, width, height);
            println!("trans: {:?}, interlace: {}", transparent_index, interlace);
            println!("delay: {}, disposal: {}, iters: {:?}",
                     frame_delay, disposal_method, self.num_iterations);
            println!("lct: {}, gct: {}", lct_flag, self.gct.is_some());
            */

            self.parse_state = Some(Box::new(create_lzw_parse_state(minimum_code_size,
                                                                    num_indices)));

            frame.indices = Some(vec![0; num_indices].into_boxed_slice());
            frame.parse_state = ParseState::MetadataParsed;
        }

        self.save_stream_position();
        Ok(())
    }

    fn parse_indices(&mut self) -> GifResult<()> {
        self.load_stream_position();
        {
            let frame = self.frames.last_mut().unwrap();
            let frame_width = frame.width as usize;
            let frame_height = frame.height as usize;
            if frame.interlace {
                let interlaced_offset = [0, 4, 2, 1];
                let interlaced_jumps = [8, 8, 4, 2];
                for i in 0..4 {
                    let mut j = interlaced_offset[i];
                    while j < frame_height {
                        try!(get_indices(self.parse_state.as_mut().unwrap(),
                                         &mut frame.indices.as_mut().unwrap()[j * frame_width..],
                                         frame_width,
                                         &mut self.data));

                        j += interlaced_jumps[i];
                    }
                }
            } else {
                try!(get_indices(self.parse_state.as_mut().unwrap(),
                                 &mut frame.indices.as_mut().unwrap(),
                                 frame_width * frame_height,
                                 &mut self.data));
            }
            frame.parse_state = ParseState::Parsed;
        }

        let frame_time = self.frames.get(self.frames.len() - 2)
                                                    .map(|frame| frame.time + frame.duration)
                                                    .unwrap_or(0);

        self.frames.push(Frame::new(frame_time));

        // TODO: save state more granularly
        self.save_stream_position();
        Ok(())
    }

    fn decode_indices(&self,
                      prev_frame_data: &mut [u8],
                      cur_frame_data: &[u8],
                      next_frame: &Frame)
        -> GifResult<()>
    {
        // ~~~~~~~~~~~~~~ INITIALIZE THE BACKGROUND ~~~~~~~~~~~

        let num_bytes = next_frame.width as usize * next_frame.height as usize * BYTES_PER_COL;

        match next_frame.disposal_method {
            // Firefox says unspecified == current
            DISPOSAL_UNSPECIFIED | DISPOSAL_CURRENT => {
                if cur_frame_data.is_empty() {
                    for byte in prev_frame_data.iter_mut() { *byte = 0 }
                } else {
                    for (output, input) in prev_frame_data.iter_mut().zip(cur_frame_data.iter()) {
                        *output = *input
                    }
                }
            }
            DISPOSAL_BG => {
                // Seems like Firefox just does this?
                for byte in prev_frame_data.iter_mut() { *byte = 0 }
                /*
                println!("BG disposal {}", self.gct_bg);
                let col_idx = self.gct_bg as usize;
                let color_map = self.gct.as_ref().unwrap();
                let is_transparent = transparent_index.map(|idx| idx as usize == col_idx)
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

                let mut buf = Vec::with_capacity(num_bytes);
                while buf.len() < num_bytes {
                    buf.push(r);
                    buf.push(g);
                    buf.push(b);
                    buf.push(a);
                }
                buf
                */
            }
            DISPOSAL_PREVIOUS => { /* no-op yay */ }
            _ => {
                // unsupported disposal method
                return Err(Error::Malformed);
            }
        };

        // nicer name
        let pixels = prev_frame_data;

        // ~~~~~~~~~~~~~~~~~~~ MAP INDICES TO COLORS ~~~~~~~~~~~~~~~~~~
        {
            let color_map: &[u8] = next_frame.lct.as_ref()
                                                 .map(|lct| &**lct)
                                                 .unwrap_or_else(|| &**self.gct.as_ref().unwrap());

            for (pix_idx, &col_idx) in next_frame.indices.as_ref().unwrap().iter().enumerate() {
                let is_transparent = next_frame.transparent_index.map(|idx| idx == col_idx)
                                                                 .unwrap_or(false);

                // A transparent pixel "shows through" to whatever pixels
                // were drawn before. True transparency can only be set
                // in the disposal phase, as far as I can tell.
                if is_transparent { continue; }

                let col_idx = col_idx as usize;
                let r = color_map[col_idx * 3 + 0];
                let g = color_map[col_idx * 3 + 1];
                let b = color_map[col_idx * 3 + 2];
                let a = 0xFF;

                // we're blitting this frame on top of some perhaps larger
                // canvas. We need to adjust accordingly.
                let offset = if next_frame.width == self.width {
                    pix_idx
                } else {
                    let row = pix_idx / next_frame.width as usize;
                    let col = pix_idx % next_frame.width as usize;
                    row * self.width as usize + col
                };

                let pix_idx = next_frame.x as usize
                            + (next_frame.y as usize * self.width as usize)
                            + offset;

                pixels[pix_idx * BYTES_PER_COL + 0] = r;
                pixels[pix_idx * BYTES_PER_COL + 1] = g;
                pixels[pix_idx * BYTES_PER_COL + 2] = b;
                pixels[pix_idx * BYTES_PER_COL + 3] = a;
            }
        }
        Ok(())
    }
}




// ~~~~~~~~~~~~~~~~~ utilities for decoding LZW data ~~~~~~~~~~~~~~~~~~~

const LZ_MAX_CODE: usize = 4095;
const LZ_BITS: usize = 12;

const NO_SUCH_CODE: usize = 4098;    // Impossible code, to signal empty.

// Stuff used in LZW decoding
struct LzwParseState {
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

fn create_lzw_parse_state(code_size: u8, pixel_count: usize) -> LzwParseState {
    let bits_per_pixel = code_size as usize;
    let clear_code = 1 << bits_per_pixel;

    LzwParseState {
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
        suffix: [0; LZ_MAX_CODE + 1],
        stack: [0; LZ_MAX_CODE],
        pixel_count: pixel_count,
    }
}

fn get_indices<R: Read>(state: &mut LzwParseState,
                        indices: &mut[u8],
                        index_count: usize,
                        data: &mut R)
    -> GifResult<()>
{
    state.pixel_count -= index_count;
    if state.pixel_count > 0xffff0000 {
        // Too much pixel data
        return Err(Error::Malformed);
    }

    try!(decompress_indices(state, indices, index_count, data));

    if state.pixel_count == 0 {
        // There might be some more data hanging around. Finish walking through
        // the data section.
        try!(skip_blocks(data));
    }

    Ok(())
}

fn decompress_indices<R: Read>(state: &mut LzwParseState,
                              indices: &mut[u8],
                              index_count: usize,
                              data: &mut R)
    -> GifResult<()>
{
    let mut i = 0;
    let mut current_prefix; // This is uninit in dgif
    let &mut LzwParseState {
        mut stack_ptr,
        eof_code,
        clear_code,
        mut last_code,
        ..
    } = state;

    if stack_ptr > LZ_MAX_CODE { return Err(Error::Malformed); }
    while stack_ptr != 0 && i < index_count {
        stack_ptr -= 1;
        indices[i] = state.stack[stack_ptr];
        i += 1;
    }

    while i < index_count {
        let current_code = try!(decompress_input(state, data));

        let &mut LzwParseState {
            ref mut prefix,
            ref mut suffix,
            ref mut stack,
            ..
        } = state;

        if current_code == eof_code { return Err(Error::Malformed); }

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
                indices[i] = current_code as u8;
                i += 1;
            } else {
                // MULTI-CODE MULTI-CODE ENGAGE -- DASH DASH DASH!!!!

                if prefix[current_code] == NO_SUCH_CODE {
                    current_prefix = last_code;

                    let code = if current_code == state.running_code - 2 {
                        last_code
                    } else {
                        current_code
                    };

                    let prefix_char = get_prefix_char(&*prefix, code, clear_code);
                    stack[stack_ptr] = prefix_char;
                    suffix[state.running_code - 2] = prefix_char;
                    stack_ptr += 1;
                } else {
                    current_prefix = current_code;
                }

                while stack_ptr < LZ_MAX_CODE
                        && current_prefix > clear_code
                        && current_prefix <= LZ_MAX_CODE {

                    stack[stack_ptr] = suffix[current_prefix];
                    stack_ptr += 1;
                    current_prefix = prefix[current_prefix];
                }

                if stack_ptr >= LZ_MAX_CODE || current_prefix > LZ_MAX_CODE {
                    return Err(Error::Malformed);
                }

                stack[stack_ptr] = current_prefix as u8;
                stack_ptr += 1;

                while stack_ptr != 0 && i < index_count {
                    stack_ptr -= 1;
                    indices[i] = stack[stack_ptr];
                    i += 1;
                }

            }

            if last_code != NO_SUCH_CODE && prefix[state.running_code - 2] == NO_SUCH_CODE {
                prefix[state.running_code - 2] = last_code;

                let code = if current_code == state.running_code - 2 {
                    last_code
                } else {
                    current_code
                };

                suffix[state.running_code - 2] = get_prefix_char(&*prefix, code, clear_code);
            }

            last_code = current_code;
        }
    }

    state.last_code = last_code;
    state.stack_ptr = stack_ptr;

    Ok(())
}

// Prefix is a virtual linked list or something.
fn get_prefix_char(prefix: &[usize], mut code: usize, clear_code: usize) -> u8 {
    let mut i = 0;

    loop {
        if code <= clear_code { break; }
        i += 1;
        if i > LZ_MAX_CODE { break; }
        if code > LZ_MAX_CODE { return NO_SUCH_CODE as u8; }
        code = prefix[code];
    }

    code as u8
}

fn decompress_input<R: Read>(state: &mut LzwParseState, src: &mut R) -> GifResult<usize> {
    let code_masks: [usize; 13] = [
        0x0000, 0x0001, 0x0003, 0x0007,
        0x000f, 0x001f, 0x003f, 0x007f,
        0x00ff, 0x01ff, 0x03ff, 0x07ff,
        0x0fff
    ];

    if state.running_bits > LZ_BITS { return Err(Error::Malformed) }

    while state.current_shift_state < state.running_bits {
        // Get the next byte, which is either in this block or the next one
        let next_byte = if state.buf[0] == 0 {

            // This block is done, get the next one
            let len = try!(read_block(src, &mut state.buf[1..]));
            state.buf[0] = len as u8;

            // Reaching the end is not expected here
            if len == 0 { return Err(Error::Malformed); }

            let next_byte = state.buf[1];
            state.buf[1] = 2;
            state.buf[0] -= 1;
            next_byte
        } else {
            // Still got bytes in this block
            let next_byte = state.buf[state.buf[1] as usize];
            // this overflows when the line is 255 bytes long, and that's ok
            state.buf[1] = state.buf[1].wrapping_add(1);
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

    Ok(code)
}


// ~~~~~~~~~~~~ Streaming reading utils ~~~~~~~~~~~~~~~

fn read_byte<R: Read>(reader: &mut R) -> GifResult<u8> {
    let mut buf = [0];
    let bytes_read = try!(reader.read(&mut buf));
    if bytes_read != 1 { return Err(Error::EndOfFile); }
    Ok(buf[0])
}

fn read_to_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> GifResult<()> {
    let mut read = 0;
    loop {
        if read == buf.len() { return Ok(()) }

        let bytes = try!(reader.read(&mut buf[read..]));

        if bytes == 0 { return Err(Error::EndOfFile) }

        read += bytes;
    }
}

/// A few places where you need to skip through some variable length region
/// without evaluating the results. This does that.
fn skip_blocks<R: Read>(reader: &mut R) -> GifResult<()> {
    let mut black_hole = [0; 255];
    loop {
        let len = try!(read_block(reader, &mut black_hole));
        if len == 0 { return Ok(()) }
    }
}

/// There are several variable length encoded regions in a GIF,
/// that look like [len, ..len]. This is a convenience for grabbing the next
/// block. Returns `len`.
fn read_block<R: Read>(reader: &mut R, buf: &mut [u8]) -> GifResult<usize> {
    debug_assert!(buf.len() >= 255);
    let len = try!(read_byte(reader)) as usize;
    if len == 0 { return Ok(0) } // read_to_full will probably freak out
    try!(read_to_full(reader, &mut buf[..len]));
    Ok(len)
}

fn le_u16(first: u8, second: u8) -> u16 {
    ((second as u16) << 8) | (first as u16)
}
