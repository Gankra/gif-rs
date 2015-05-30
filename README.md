# gif-rs
Eats a gif, spits out fully decoded frames. 

This is the first step towards full animated gif support in Servo and rust-media.
Obviously work needs to be done to make it work in a more incremental manner.
It also currently does much more allocation and work than is probably strictly necessary.

A utility for testing out that output is correct is provided in the examples dir. It tries to
load the first argument as a path to a gif, decodes it, and then spits out the frames as 
individual `.tga` images in an `output/` folder. The data folder comes with some samples you can run.

You can run it with
```
> cargo run --release --example test data/423tribalChallenge.gif
```
