extern crate gif;

use std::fs::{self, File};
use std::io::Result as IoResult;
use std::io::Write;
use std::env;

static OUT_DIR: &'static str = "output";

fn main() {
    let mut args = env::args();
    args.next();

    let path = args.next().unwrap();

	if let Err(err) = do_it(&path) {
        println!("OH NO! {}", err);
    }
}

fn do_it(path: &str) -> IoResult<()> {
	let file = try!(File::open(path));

	let gif = try!(gif::parse_gif(&file));

    if fs::read_dir(OUT_DIR).is_ok() {
        try!(fs::remove_dir_all(OUT_DIR));
    }
    try!(fs::create_dir(OUT_DIR));

    for (idx, frame) in gif.frames.iter().enumerate() {
        let image = RBitmap {
            width: gif.width,
            height: gif.height,
            data: &frame.data,
        };
        try!(save(&image, idx));
    }

	Ok(())
}

struct RBitmap<'a> {
	width: u16,
	height: u16,
	data: &'a [u8],
}

fn save<'a>(data: &RBitmap<'a>, num: usize) -> IoResult<()> {
    let file_name = format!("{}/frame-{:03}.tga", OUT_DIR, num);
	let mut file = try!(File::create(&file_name));

	let mut header = [0; 18];
    header[2] = 2; // truecolor
    header[12] = data.width as u8 & 0xFF;
    header[13] = (data.width >> 8) as u8 & 0xFF;
    header[14] = data.height as u8 & 0xFF;
    header[15] = (data.height >> 8) as u8 & 0xFF;
    header[16] = 32; // bits per pixel

    try!(file.write_all(&header));

    // The image data is stored bottom-to-top, left-to-right
    for y in (0..data.height).rev() {
        for x in 0..data.width {
            let idx = (x as usize + y as usize * data.width as usize) * 4;
            let r = data.data[idx + 0];
            let g = data.data[idx + 1];
            let b = data.data[idx + 2];
            let a = data.data[idx + 3];
            try!(file.write_all(&[b, g, r, a]));
        }
    }


    // The file footer
    let footer = b"\0\0\0\0\0\0\0\0TRUEVISION-XFILE.\0";

    try!(file.write_all(footer));

    Ok(())
}