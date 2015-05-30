extern crate gif;

use std::fs::File;
use std::io::Result as IoResult;

fn main() {
	do_it().unwrap();
}

fn do_it() -> IoResult<()> {
	let file = File::new("~/Downloads/sample_1.gif");
	println!("{:?}", file);
}