//ChatGPT assisted me with this.
//Base code came from https://stackoverflow.com/questions/72181090/getting-pixels-of-many-different-colors-when-drawing-with-sdl-in-rust

extern crate sdl2;

use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use std::sync::{Arc, Mutex};
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use anyhow;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::thread;
//use std::env;
use clap::Parser;
use std::collections::VecDeque;

mod viscolors;

const WINDOW_WIDTH: i32 = 75;
const WINDOW_HEIGHT: i32 = 16;
/* const NUM_BARS: usize = 75;
const DECAY_RATE: f64 = 0.0003; // Adjust the decay rate as needed

static mut LEVEL2: f32 = 15.0;
static mut PEAK1: f64 = 0.0; */
//static mut BINS: usize = 0;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Oscilloscope style
    #[arg(short, long, default_value = "lines")]
    oscstyle: String, // Change this to String

    /// Spectrum Analyzer style
    #[arg(short, long, default_value = "normal")]
    specdraw: String, // Change this to String

    /// Name of the custom viscolor.txt file
    #[arg(short, long, default_value = "viscolor.txt")]
    viscolor: String,
    
    /// Index of the audio device to use
    #[arg(short, long)]
    device: Option<usize>,

    /// Zoom factor
    #[arg(short, long, default_value = "7")]
    zoom: i32,

    /// Amplify the incoming signal
    #[arg(short, long, default_value = "1.0")]
    amp: f32,

    /// Specify the visualization mode to use
    #[arg(short, long, default_value = "0")]
    mode: u8,
}

/* #[derive(Clone)]
struct Bar {
    height: f64,
    peak: f64,
    gravity: f64,
}

impl Bar {
    fn new() -> Self {
        Self {
            height: 0.0,
            peak: 0.0,
            gravity: 0.2,
        }
    }
} */

fn hamming_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.54 - 0.46 * f32::cos(2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32))
        .collect()
}

fn switch_oscstyle(oscstyle: &mut &str) {
    match *oscstyle {
        "dots" => *oscstyle = "lines",
        "lines" => *oscstyle = "solid",
        "solid" => *oscstyle = "dots",
        _ => println!("Invalid oscilloscope style. Supported styles: dots, lines, solid."),
    }
}

fn switch_specstyle(specdraw: &mut &str) {
    match *specdraw {
        "normal" => *specdraw = "fire",
        "fire" => *specdraw = "line",
        "line" => *specdraw = "normal",
        _ => println!("Invalid analyzer style. Supported styles: normal, fire, line."),
    }
}

fn draw_oscilloscope(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    _colors: &[Color],
    osc_colors: &[Color],
    ys: &[u8],
    fft: &[u8],
    oscstyle: &str,
    specdraw: &str,
    mode: u8,
    zoom: i32,
) {
    let xs: Vec<i32> = (0..WINDOW_WIDTH).collect();
    let ys: Vec<i32> = ys.iter().step_by(16).map(|&sample| ((sample as i32 / 8) - 9)/* * WINDOW_HEIGHT / 16*/).collect(); // cast to i32
    let fft: Vec<i32> = fft.iter().step_by(11).map(|&sample| ((sample as i32 / 8) - 9)).collect(); // cast to i32

    //let mut bars: Vec<Bar> = vec![Bar::new(); NUM_BARS];

    // Process your FFT data and store it in the respective Bar instances
/* unsafe {
    // Iterate over all bars to process FFT data and update peaks.
    for (bar, &fft_value) in bars.iter_mut().zip(fft.iter()) {
        // Access the values from bars and fft here, which are represented by (bar, fft_value).
        // You can use 'bar' and 'fft_value' as needed in this section of the loop.
    
        // For example, if you want to update 'bar.height' based on 'fft_value':
        bar.height = fft_value as f64 + 9.0;
    
        // Or if you want to modify 'fft_value' and update 'bar.height':
        // fft_value = some_modification(fft_value);
        // bar.height = fft_value;

        //println!("{}", bar.height+9.0);
        /*if bar.height > bar.peak {
            bar.peak = bar.height;
        } else {
            bar.peak -= 0.001;
        } if bar.peak >= 15.0{
            bar.peak = 15.0;
        } if bar.peak < 1.0 {
            bar.peak = -1.0;
        }
        //bar.peak = PEAK1;
        if mode == 0 {
            //println!("{}", bar.peak);
        }*/
    }
} */
/* let mut bins: usize = 0;
    while bins < 75{
    // Decrease bars[bins].peak by 3.0

    // Ensure bars[bins].peak doesn't go below a certain minimum value (e.g., 0.0)
        if bars[bins].height >= bars[bins].peak {
            //bars[bins].peak = bars[bins].height;
            bars[bins].peak = bars[bins].height;
        } if bars[bins].peak >= 15.0{
            bars[bins].peak = 15.0;
        } if bars[bins].peak < 1.0 {
            bars[bins].peak = -1.0;
        } else {
            //bars[bins].peak -= 0.2;
        }
        //bars[bins].height = bars[bins].peak;

        println!("{:?}", bars[bins].peak);
        bins += 1;
    } */

    let mut last_y = 0;
    let mut top: i32 = 0; //bro it is being read though wth?!
    let mut bottom: i32 = 0;

    for x in 0..WINDOW_WIDTH {
        for y in 0..WINDOW_HEIGHT {
            if x % 2 == 1 || y % 2 == 0 {
                let rect = Rect::new(x * zoom, y * zoom, zoom as u32, zoom as u32);
                canvas.set_draw_color(_colors[0]);
                canvas.fill_rect(rect).unwrap();
            } else {
                let rect = Rect::new(x * zoom, y * zoom, zoom as u32, zoom as u32);
                canvas.set_draw_color(_colors[1]);
                canvas.fill_rect(rect).unwrap();
            }
        }
    }
/* if mode == 0 {
    unsafe {
        for (index, bar) in bars.iter_mut().enumerate() {
            // Perform your peak update logic (similar to what you've described earlier)
            // For demonstration purposes, we'll just set peak1 to a random value.
            //bar.peak = rand::random::<f64>() * 100.0;

            // Calculate the bar's position and height based on peak
            let bar_x = (index as i32 * (WINDOW_WIDTH / NUM_BARS as i32)) as i32;
            let bar_height = (bar.peak) as i32;

            // Create and draw the bar as a rectangle
            let rect = Rect::new(bar_x * zoom, (-bar_height as i32 +15) * zoom as i32, zoom as u32, zoom as u32);
            canvas.set_draw_color(_colors[23]); // Set your desired color
            canvas.fill_rect(rect).unwrap();
        }
    }
}
*/
    if mode == 1{
        for (x, y) in xs.iter().zip(ys.iter()) {
            let x = *x;
            let y = *y;
    
            let x = std::cmp::min(std::cmp::max(x, 0), 75 - 1);
            let y = std::cmp::min(std::cmp::max(y, 0), 16 - 1);
    
            if x == 0 {
                last_y = y;
            }
    
            top = y;
            bottom = last_y;
            last_y = y;
    
            if oscstyle == "lines" {
                if bottom < top {
                    std::mem::swap(&mut bottom, &mut top);
                    top += 1;
                }
        
                for dy in top..=bottom {
                    let color_index = (top as usize) % osc_colors.len();
                    let scope_color = osc_colors[color_index];
                    let rect = Rect::new(x * zoom, dy * zoom, zoom as u32, zoom as u32);
                    canvas.set_draw_color(scope_color);
                    canvas.fill_rect(rect).unwrap();
                }
            } else if oscstyle == "solid" {
                if y >= 8{
                    top = 8;
                    bottom = y;
                } else {
                    top = y;
                    bottom = 7;
                }
    
                for dy in top..=bottom {
                    let color_index = (y as usize) % osc_colors.len();
                    let scope_color = osc_colors[color_index];
                    let rect = Rect::new(x * zoom, dy * zoom, zoom as u32, zoom as u32);
                    canvas.set_draw_color(scope_color);
                    canvas.fill_rect(rect).unwrap();
                }
            } else if oscstyle == "dots" {
                for _dy in -1..y {
                    let color_index = (y as usize) % osc_colors.len();
                    let scope_color = osc_colors[color_index];
                    let rect = Rect::new(x * zoom, y * zoom, zoom as u32, zoom as u32);
                    canvas.set_draw_color(scope_color);
                    canvas.fill_rect(rect).unwrap();
                }
            } else {
                eprintln!("Invalid oscilloscope style. Supported styles: lines, solid, dots.");
                // You can handle this error case according to your needs
                // ...
            }
        }

    } else if mode == 0{
        //let ys: Vec<i32> = vec![156246, 100569, 28685, 32621, 28655, 14996, 17003, 1479, 11935, 5861, 7520, 8576, 3010, 7934, 1077, 5509, 4701, 5180, 5782, 3093, 4452, 436, 3471, 2699, 2558, 4720, 717, 5868, 2421, 2884, 1485, 2499, 1833, 675, 1918, 924, 1676, 3008, 2078, 2807, 706, 2600, 738, 2264, 1311, 1586, 1008, 356, 1813, 855, 1640, 1642, 929, 1759, 244, 1682, 717, 1320, 1554, 625, 1525, 71, 1487, 969, 1851, 3035, 1552, 1978, 461, 1660, 879, 1070, 1092, 1733, 3462, 1137, 1269, 685, 1173, 1183, 453, 1196, 303, 1080, 770, 975, 1069, 356, 1037, 179, 1074, 694, 825, 1022, 246, 959, 185, 898, 727, 692, 1024, 234, 845, 258, 843, 708, 594, 940, 139, 764, 302, 792, 705, 521, 874, 64, 711, 323, 733, 707, 458, 828, 29, 642, 350, 660, 708, 376, 793, 47, 643, 387, 622, 683, 329, 750, 77, 582, 380, 526, 686, 248, 716, 126, 601, 398, 499, 674, 200, 684, 124, 570, 433, 494, 645, 181, 637, 80, 447, 468, 401, 664, 153, 577, 200, 502, 486, 413, 600, 83, 560, 185, 459, 496, 367, 583, 50, 521, 219, 474, 461, 302, 578, 42, 436, 189, 420, 455, 248, 545, 67, 455, 260, 390, 521, 266, 529, 48, 391, 265, 392, 457, 176, 515, 58, 372, 274, 349, 477, 156, 497, 84, 349, 340, 362, 475, 127, 499, 82, 363, 300, 324, 435, 87, 452, 106, 341, 320, 266, 474, 71, 442, 138, 363, 320, 253, 465, 64, 410, 122, 303, 350, 227, 453, 42, 378, 152, 318, 322, 172, 448, 31, 359, 183, 288, 374, 179, 435, 31, 321, 181, 275, 363, 138, 435, 60, 323, 193, 241, 397, 134, 422, 39, 293, 218, 256, 363, 104, 395, 60, 261, 248, 251, 365, 93, 368, 72, 249, 242, 217, 367, 70, 354, 109, 258, 243, 177, 375, 39, 334, 113, 253, 265, 183, 360, 17, 321, 122, 248, 262, 147, 369, 23, 323, 119, 209, 294, 142, 345, 40, 315, 130, 210, 297, 112, 371, 29, 293, 117, 175, 321, 109, 356, 45, 281, 168, 207, 308, 112, 330, 32, 218, 191, 199, 297, 70, 333, 78, 213, 191, 179, 320, 82, 302, 72, 211, 194, 159, 298, 39, 298, 84, 194, 209, 138, 310, 19, 290, 97, 190, 208, 112, 317, 34, 288, 99, 187, 228, 135, 289, 5, 255, 105, 155, 249, 106, 300, 35, 248, 129, 165, 262, 112, 290, 54, 221, 114, 132, 268, 69, 311, 38, 211, 135, 151, 243, 52, 275, 65, 201, 143, 115, 266, 36, 301, 68, 196, 157, 130, 265, 29, 286, 80, 196, 170, 120, 270, 25, 250, 96, 164, 195, 107, 282, 27, 256, 90, 161, 194, 86, 283, 22, 277, 85, 146, 183, 65, 285, 33, 253, 87, 131, 196, 54, 291, 24, 229, 102, 118, 235, 82, 245, 51, 209, 119, 145, 210, 56, 256, 39, 177, 129, 104, 231, 34, 264, 70, 190, 145, 131, 222, 35, 235, 74, 166, 164, 111, 242, 15, 253, 76, 158, 145, 96, 240, 4, 254, 73, 155, 168, 76, 248, 11, 237, 91, 138, 180, 91, 241, 18, 219, 103, 150, 183, 67, 237, 27, 185, 127, 132, 198, 61, 253, 28, 204, 101, 104, 205, 55, 235, 43, 184, 113, 109, 181, 25, 246, 49, 162, 125, 102, 207, 34, 224, 76, 167, 137, 92, 196, 29, 207, 89, 161, 126, 88, 202, 21, 211, 95, 158, 134, 73, 204, 38, 229, 76, 147, 140, 60, 204, 38, 187, 116, 130, 164, 56, 228, 28, 200, 105, 103, 153, 46, 225, 41, 203, 111, 118, 172, 50, 217, 38, 179, 107, 89, 177, 33, 225, 56, 176, 123, 97, 170, 28, 209, 69, 179, 115, 98, 170, 26, 198, 71, 143, 132, 84, 193, 26, 223, 61, 146, 121, 63, 194, 16, 192, 80, 113, 132, 52, 183, 28, 194, 96, 123, 144, 48, 215, 15, 187, 94, 111, 138, 58, 194, 45, 179, 115, 105, 143, 39, 199, 47, 180, 110, 103, 140, 13, 207, 58, 172, 111, 78, 156, 25, 195, 69, 171, 116, 87, 152, 11, 195, 76, 153, 122, 72, 158, 11, 199, 94, 164, 114, 63, 176, 20, 211, 71, 131, 124, 55, 182, 14, 182, 94, 121, 132, 55, 176, 35, 190, 106, 125, 126, 48, 185, 29, 180, 110, 125, 126, 42, 189, 42, 178, 111, 107, 123, 48, 194, 47, 172, 111, 107, 142, 31, 179, 60, 155, 122, 97, 145, 12, 178, 71, 144, 134, 94, 140, 20, 195, 86, 156, 117, 57, 154, 22, 197, 73, 135, 129, 61, 160, 28, 176, 88, 135, 115, 52, 160, 31, 187, 88, 115, 127, 47, 169, 28, 177, 100, 121, 113, 37, 158, 42, 175, 92, 96, 118, 32, 151, 48, 139, 128, 92, 120, 11, 171, 66, 161, 120, 82, 145, 20, 179, 54, 150, 122, 86, 117, 32, 185, 68, 147, 125, 76, 123, 24, 183, 59, 126, 116, 52, 149, 8, 150, 84, 117, 130, 61, 162, 18, 159, 99, 113, 131, 42, 155, 33, 169, 98, 106, 136, 38, 159, 37, 156, 112, 106, 122, 29, 164, 42, 160, 118, 111, 118, 25, 155, 69, 167, 112, 85, 135, 25, 149, 65, 142, 128, 83, 123, 12, 165, 72, 146, 135, 95, 121, 15, 169, 72, 144, 114, 65, 133, 18, 164, 85, 132, 118, 43, 128, 19, 155, 96, 122, 129, 47, 137, 22, 149, 100, 96, 150, 36, 153, 22, 148, 112, 111, 141, 47, 133, 55, 157, 117, 109, 132, 35, 132, 58, 146, 115, 84, 134, 18, 132, 77, 141, 133, 85, 127, 5, 154, 80, 150, 128, 70, 143, 15, 148, 77, 138, 136, 80, 122, 9, 152, 81, 121, 141, 51, 146, 27, 175, 88, 145, 128, 64, 133, 43, 90, 88, 81, 144, 53, 123, 52, 113, 185, 145, 158, 46, 164, 6, 153, 91, 106, 127, 33, 139, 62, 158, 116, 96, 150, 31, 151, 28, 125, 118, 82, 144, 34, 126, 73, 139, 123, 81, 128, 14, 87, 122, 141, 147, 73, 162, 47, 108, 124, 136, 163, 99, 103, 11, 125, 126, 170, 102, 46, 133, 15, 138, 102, 117, 154, 66, 127, 39, 152, 88, 106, 133, 31];
        //debug array until i figure out FFT
        for (x, y) in xs.iter().zip(fft.iter()) {
            let x = *x;
            let y = -*y+6;
    
            let x = std::cmp::min(std::cmp::max(x, 0), 75 - 1);
            let y = std::cmp::min(std::cmp::max(y, 0), 16 - 1);

            top = y; //come on now :|
            bottom = 16;

            if y >= 16{
                top = 17;
                bottom = y;
            } else {
                top = y + 1;
                bottom = 16;
            }

            for dy in top..=bottom {
                let color_index: usize;
                if specdraw == "normal"{
                    color_index = (dy as usize + 2) % _colors.len();
                } else if specdraw == "fire" {
                    color_index = (dy as usize - y as usize + 2) % _colors.len();
                } else if specdraw == "line" {
                    if y == -1 {
                        color_index = 2;
                    } else {
                        color_index = (2u32).wrapping_add(y as u32) as usize % _colors.len();
                    } 
                } else {
                    color_index = 0;
                }
                let color = _colors[color_index];
                let rect = Rect::new(x * zoom, dy * zoom, zoom as u32, zoom as u32);
                canvas.set_draw_color(color);
                canvas.fill_rect(rect).unwrap();
            }
        }
    } else if mode == 2{
    }
}

fn audio_stream_loop(tx: Sender<Vec<u8>>, s: Sender<Vec<u8>>, selected_device_index: Option<usize>, amp: f32) {
    enum ConfigType {
        WindowsOutput(cpal::SupportedStreamConfig),
        WindowsInput(cpal::SupportedStreamConfig),
        Unix(cpal::StreamConfig),
        Darwin(cpal::SupportedStreamConfig),
    }
    let host = cpal::default_host();
    let device = match selected_device_index {
        Some(index) => {
            let mut devices = host.devices().expect("Failed to retrieve devices");
            let device = devices.nth(index).expect("Invalid device index.");
            device
        }
        None => todo!(), // Consider a proper fallback here
    };
    let config: ConfigType;

    if cfg!(windows) {
        let supports_input = device.supported_input_configs().is_ok();
        if supports_input {
            let input_config = device.default_input_config();
            match input_config {
                Ok(conf) => {
                    config = ConfigType::WindowsInput(conf);
                }
                Err(_) => {
                    // Fallback to output stream if input config is not supported
                    config = ConfigType::WindowsOutput(device.default_output_config().expect("Failed to get default output config"));
                }
            }
        } else {
            config = ConfigType::WindowsOutput(device.default_output_config().expect("Failed to get default output config"));
        }
    } else if cfg!(unix) {
        config = ConfigType::Unix(cpal::StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(44100),
            buffer_size: cpal::BufferSize::Fixed(2048),
        });
    } else if cfg!(macos) {
        config = ConfigType::Darwin(device.default_output_config().expect("Failed to get default output config"));
    } else {
        panic!("Unsupported platform");
    }

    // ring buffer (VecDeque)
    let mut ring_buffer: VecDeque<u8> = VecDeque::with_capacity(2048); //HAHA SCREW YOU WASAPI, NOW YOU WILL NOT COMPLAIN

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Convert f32 samples to u8 (0-255) and collect them into a Vec<u8>
        let left: Vec<f32> = data
            .iter()
            /* .step_by(16) */ // Skip every other sample (right channel)
            .map(|&sample| (((-sample * amp)) * 127.5))
            .collect();

        let right: Vec<f32> = data
            .iter()
            .skip(1)
            /* .step_by(16) */ // Skip every other sample (right channel)
            .map(|&sample| (((-sample * amp)) * 127.5))
            .collect();

        let mixed: Vec<u8> = left
            .iter()
            .zip(right.iter())
            .map(|(left_sample, right_sample)| (((*left_sample as f32 + *right_sample as f32 + 256.0) / 2.0) + 5.0) as u8)
            .collect();

        let mixed_fft: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(left_sample, right_sample)| (((*left_sample + *right_sample) / 2.0)))
            .collect();

        // Extend the ring buffer with the new samples
        for left_sample in &mixed {
            if ring_buffer.len() == ring_buffer.capacity() {
                ring_buffer.pop_front(); // Remove the oldest sample when the buffer is full
            }
            ring_buffer.push_back(*left_sample);
        }

        // Apply the Hamming window to mixed_fft
        let window = hamming_window(mixed_fft.len());

        let windowed_mixed_fft: Vec<f32> = mixed_fft
            .iter()
            .zip(window.iter())
            .map(|(&sample, &window_sample)| sample * window_sample / 12.0)
            .collect();

        // Convert mixed to a [f32; 16] array
        let mut mixed_f32: [f32; 4096] = [0.0; 4096];
        for (i, &sample) in windowed_mixed_fft.iter().enumerate() {
            mixed_f32[i] = sample as f32;
        }

        // compute the RFFT of the samples
        //let mut mixed_f32: [f32; 2048] = mixed.try_into().unwrap();
        let spectrum = microfft::real::rfft_4096(&mut mixed_f32);
        // since the real-valued coefficient at the Nyquist frequency is packed into the
        // imaginary part of the DC bin, it must be cleared before computing the amplitudes
        //spectrum[0].im = 0.0;

        let amplitudes: Vec<_> = spectrum.iter().map(|c| c.l1_norm() as u8).collect();
        //println!("{amplitudes:?}");
        //assert_eq!(&amplitudes, &[0, 0, 0, 8, 0, 0, 0, 0]);

        // Convert the ring buffer to a regular Vec<u8> and send it through the channel
        tx.send(ring_buffer.iter().copied().collect()).unwrap();
        s.send(amplitudes.iter().copied().collect()).unwrap();
    };

    // When creating the stream, pattern match on the ConfigType to get the appropriate config
    let stream = match config {
        ConfigType::WindowsOutput(conf) | ConfigType::WindowsInput(conf) => {
            device.build_input_stream(&conf.into(), callback, err_fn, None)
        }
        ConfigType::Unix(conf) => device.build_input_stream(&conf.into(), callback, err_fn, None),
        ConfigType::Darwin(conf) => device.build_input_stream(&conf.into(), callback, err_fn, None),
    }
    .unwrap();    
    stream.play().unwrap();

    // The audio stream loop should not block, so we use an empty loop.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(32768));
    }
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let selected_device_index = match args.device {
        Some(index) => {
            if index == 0 {
                eprintln!("Device index should start from 1.");
                std::process::exit(1);
            }
            index - 1
        }
        None => {
            // Prompt the user to select an audio device
            let selected_device_index = prompt_for_device();
            if selected_device_index.is_none() {
                std::process::exit(1);
            }
            selected_device_index.unwrap()
        }
    };
    
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    //let args: Vec<String> = env::args().collect();

    // Extract the oscstyle field from Args struct
    let mut oscstyle = args.oscstyle.as_str(); // Convert String to &str
    let mut specdraw = args.specdraw.as_str();
    let zoom = args.zoom;
    let amp = args.amp;
    let window = video_subsystem
        .window("Winamp Mini Visualizer (in Rust)", (WINDOW_WIDTH * zoom) as u32, (WINDOW_HEIGHT * zoom) as u32)
        .position_centered()
        .build()
        .unwrap();

    // Load the custom viscolor.txt file
    let mut viscolors = viscolors::load_colors(&args.viscolor);
    let mut osc_colors = osc_colors_and_peak(&viscolors);
    let mut mode = args.mode;

    let mut canvas = window.into_canvas().build().unwrap();

    let mut event_pump = sdl_context.event_pump().unwrap();

    // The vector to store captured audio samples.
    let audio_data = Arc::new(Mutex::new(Vec::<u8>::new()));
    let spec_data = Arc::new(Mutex::new(Vec::<u8>::new()));

    // Create a blocking receiver to get audio samples from the audio stream loop.
    let (tx, rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = unbounded();
    let (s, r): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = unbounded();

    // Start the audio stream loop in a separate thread.
    thread::spawn(move || audio_stream_loop(tx, s, Some(selected_device_index), amp));

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown { keycode: Some(Keycode::R), .. } => {
                    // Reload the viscolors data when "R" key is pressed
                    let new_viscolors = viscolors::load_colors(&args.viscolor);
                    let new_osc_colors = osc_colors_and_peak(&new_viscolors);
                    viscolors = new_viscolors;
                    osc_colors = new_osc_colors;
                }
                Event::MouseButtonDown { mouse_btn: MouseButton::Right, .. } => {
                    if mode == 1{
                        switch_oscstyle(&mut oscstyle);
                    } else if mode == 0 {
                        switch_specstyle(&mut specdraw);
                    }
                }
                Event::MouseButtonDown { mouse_btn: MouseButton::Left, .. } => {
                    mode = (mode + 1) % 3;
                    //println!("{mode}")
                }
                _ => {}
            }
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        // Lock the mutex and swap the captured audio samples with the visualization data.
        let audio_samples = rx.recv().unwrap();
        let spec_samples = r.recv().unwrap();
        //println!("Captured audio samples: {:?}", audio_samples);

        // Lock the mutex and update the captured audio samples.
        let mut audio_data = audio_data.lock().unwrap();
        *audio_data = audio_samples;

        let mut spec_data = spec_data.lock().unwrap();
        *spec_data = spec_samples;
        //println!("Captured audio samples: {:?}", audio_data);

        draw_oscilloscope(&mut canvas, &viscolors, &osc_colors, &*audio_data, &*spec_data, oscstyle, specdraw, mode, zoom/* , modern*/);

        canvas.present();

        std::thread::sleep(std::time::Duration::from_millis(0));
    }

    // Stop the audio streaming loop gracefully
    //audio_thread.join().unwrap();
    //gracefully my ass ChatGPT, this shit hung the entire thing on closing
    Ok(())
}

fn prompt_for_device() -> Option<usize> {
    let host = cpal::default_host();
    let devices = host.devices().expect("Failed to retrieve devices").collect::<Vec<_>>();
    
    println!("Available audio devices:");
    for (index, device) in devices.iter().enumerate() {
        println!("{}. {}", index + 1, device.name().unwrap_or("Unknown Device".to_string()));
    }
    
    println!("Enter the number of the audio device (Speakers or Microphone) to visualize: ");

    loop {
        //println!("Please select an audio device (1 - {}):", devices.len());
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).expect("Failed to read line");

        if let Ok(index) = input.trim().parse::<usize>() {
            if index > 0 && index <= devices.len() {
                return Some(index - 1); // Convert to 0-based index
            } else {
                println!("Invalid input.");
                println!("Please select an audio device (1 - {}):", devices.len());
            }
        }
    }
}

fn osc_colors_and_peak(colors: &[Color]) -> Vec<Color> {
    if colors.len() == 35 {
        vec![
            colors[18],
            colors[19],
            colors[20],
            colors[21],
            colors[22],
            colors[23],
            colors[24],
            colors[25],
            colors[26],
            colors[27],
            colors[28],
            colors[29],
            colors[30],
            colors[31],
            colors[32],
            colors[33],
        ]
    } else {
        vec![
            colors[21],
            colors[21],
            colors[20],
            colors[20],
            colors[19],
            colors[19],
            colors[18],
            colors[18],
            colors[19],
            colors[19],
            colors[20],
            colors[20],
            colors[21],
            colors[21],
            colors[22],
            colors[22],
        ]
    }
}