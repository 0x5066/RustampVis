//ChatGPT assisted me with this.
//Base code came from https://stackoverflow.com/questions/72181090/getting-pixels-of-many-different-colors-when-drawing-with-sdl-in-rust

extern crate sdl2;

use clap::Parser;

use sdl2::image::{InitFlag, LoadSurface};
use sdl2::keyboard::Keycode;
use sdl2::mouse::Cursor;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::event::{Event, WindowEvent};
use sdl2::video::WindowContext;
use sdl2::mouse::MouseButton;
use sdl2::render::TextureCreator;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{unbounded, Receiver, Sender};

use num::Complex;
use num::complex::ComplexFloat;

use std::thread;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use image::GenericImageView;

mod viscolors;
mod comctl;

use comctl::listview_box;
use comctl::groupbox;
use comctl::render_text;
use comctl::newline_handler;
use comctl::draw_dropdown;
use comctl::checkbox;
use comctl::slider_small;
use comctl::tab;
use comctl::button;
use comctl::radiobutton;

const WINDOW_WIDTH: i32 = 75;
const WINDOW_HEIGHT: i32 = 16;
const NUM_BARS: usize = 75;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Oscilloscope style
    #[arg(short, long, default_value = "lines")]
    oscstyle: String, // Change this to String

    /// Spectrum Analyzer style
    #[arg(short, long, default_value = "normal")]
    specdraw: String, // Change this to String

    /// Name of the custom viscolor.txt file, supports images in the viscolor.txt format as well.
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

    /// Bandwidth of the Analyzer
    #[arg(short, long, default_value = "thick")]
    bandwidth: String,

    /* /// Modern Skin style visualization
    #[arg(short, long, default_value = "0")]
    modern: bool,*/

    /// Set peak fall off, ranging from 1 - 5
    #[arg(long, default_value = "3")]
    peakfo: u8,

    /// Set analyzer fall off, ranging from 1 - 5
    #[arg(long, default_value = "2")]
    barfo: u8,

    /// Enable/Disable peaks
    #[arg(long, default_value = "1")]
    peaks: u8,
}

#[derive(Copy)]
#[derive(Clone)]
struct Bar {
    height: f64,
    height2: f64,
    peak: f64,
    gravity: f64,
    bargrav: f64
}

fn hamming_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.54 - 0.46 * f32::cos(2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32))
        .collect()
}

// Define your A-weighting values and frequency values here
const A_WEIGHTING: [f32; 29] = [
    // 20   25    31,5     40    50    63     80    100  125  160  200  250  315  400
    -12.0, -12.0, -12.0, -12.0, -8.0, -7.0, -6.0, -5.0, -4.0, -2.0, 2.0, 4.5, 6.0, 8.5,
    //500 630 800  1000 1250 1600 2000 2500 3150  4000  5000  6300  8000  10000 16000
    12.0, 12.5, 14.0, 14.5, 16.0, 16.5, 18.0, 18.5, 20.0, 22.5, 24.0, 24.5, 28.0, 28.5, 32.0
];

const F_VALUES: [f32; 29] = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0,
    400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
    5000.0, 6300.0, 8000.0, 10000.0, 16000.0,
];

fn apply_weighting(spectrum: &mut [Complex<f32>], frequencies: &[f32]) {
    //let mut weights = vec![1.0; spectrum.len()];

    for (i, &spectrum_freq) in frequencies.iter().enumerate() {
        let closest_index = F_VALUES
            .iter()
            .map(|&f| (spectrum_freq - f).abs())
            .enumerate()
            .min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Apply the weight to the spectrum value at the closest_index
        let weight = 10.0_f32.powf(A_WEIGHTING[closest_index] / 20.0);
        spectrum[i] = spectrum[i] * Complex::new(weight, 0.0); // Apply the weight
    }
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

fn switch_bandwidth(bandwidth: &mut &str){
    match *bandwidth {
        "thick" => *bandwidth = "thin",
        "thin" => *bandwidth = "thick",
        _ => println!("Invalid bandwidth. Supported bandwidths: thick, thin."),
    }
}

fn linear_interpolation(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    // Ensure x0 is less than x1
    let (x0, x1, y0, y1) = if x0 > x1 {
        (x1, x0, y1, y0)
    } else {
        (x0, x1, y0, y1)
    };

    // Calculate the interpolated value
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

fn process_fft_data(bars: &mut [Bar], fft_iter: &mut std::slice::Iter<f64>, bandwidth: &str, peakfo: u8) {
    if bandwidth == "thick" {
        for bars_chunk in bars.chunks_mut(4) {
            let mut sum = 0.0;

            for _ in 0..24 * 2 {
                if let Some(fft_value) = fft_iter.next() {
                    sum += *fft_value as f64 + 9.0;
                } else {
                    break;
                }
            }

            for bar in bars_chunk.iter_mut().take(4) {
                bar.height = sum / (25.0 * 2.0);
                if bar.height >= 15.0 {
                    bar.height = 15.0;
                }
            }
        }
    } else {
        for bars_chunk in bars.chunks_mut(1) {
            let mut sum = 0.0;

            for _ in 0..6 * 2 {
                if let Some(fft_value) = fft_iter.next() {
                    sum += *fft_value as f64 + 9.0;
                } else {
                    break;
                }
            }

            for bar in bars_chunk.iter_mut() {
                bar.height = sum / (7.0 * 2.0);
                if bar.height >= 15.0 {
                    bar.height = 15.0;
                }
            }
        }
    }

    for i in 0..NUM_BARS {
        bars[i].height2 -= bars[i].bargrav;

        if bars[i].height2 <= bars[i].height {
            bars[i].height2 = bars[i].height;
        }
        if bars[i].height2 > bars[i].peak {
            bars[i].gravity = 0.0;
            bars[i].peak = bars[i].height2;
        } else {
            if bars[i].gravity <= 16.0 {
                bars[i].gravity += (1.0 / 512.0) * (peakfo as f64);
            }
            bars[i].peak = if bars[i].peak <= 0.0 {
                0.0
            } else {
                bars[i].peak - bars[i].gravity
            };
        }
    }
}

fn draw_visualizer(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    _colors: &[Color],
    osc_colors: &[Color],
    peak_color: Color,
    ys: &[u8],
    fft: &[u8],
    oscstyle: &str,
    specdraw: &str,
    mode: u8,
    bandwidth: &str,
    zoom: i32,
    bars: &mut [Bar],
    peakfo: u8,
    peaks: Arc<Mutex<u8>>,
) {
    let peaks_unlocked = peaks.lock().unwrap();
    let xs: Vec<i32> = (0..75).collect();
    let ys: Vec<i32> = ys.iter().step_by(16).map(|&sample| ((sample as i32 / 8) - 9)/* * WINDOW_HEIGHT / 16*/).collect(); // cast to i32
    let fft: Vec<f64> = fft.iter()
    .map(|&sample| ((sample as i32 / 8) - 9) as f64)
    .collect(); // cast to i32

    let mut last_y = 0;
    let mut top: i32;
    let mut bottom: i32;

    // analyzer stuff
    process_fft_data(bars, &mut fft.iter(), bandwidth, peakfo);

    for x in 0..75 {
        for y in 0..16 {
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
            }
        }

    } else if mode == 0{
        for (i, bar) in bars.iter().enumerate() {
            let x = i as i32 * zoom as i32;
            let y = -bar.height2 as i32 + 15;
            
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
                    color_index = (dy as usize).wrapping_sub(y as usize).wrapping_add(2) % _colors.len();
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
                let rect = Rect::new(x, dy * zoom, zoom as u32, zoom as u32);
                canvas.set_draw_color(color);
                canvas.fill_rect(rect).unwrap();
            }
        }
        for (i, bar) in bars.iter().enumerate() {
            let bar_x = i as i32 * zoom as i32;
            let bar_height = -bar.peak + 15.98999;
            let mut peaki32: i32 = bar_height as i32;

        if peaki32 > 14 {
            peaki32 = 18;
        }

            let rect = Rect::new(
                bar_x, peaki32 * zoom as i32,
                zoom as u32,
                zoom as u32,
            );
            let color = peak_color;
            canvas.set_draw_color(color);
            if *peaks_unlocked == 1 {
                canvas.fill_rect(rect).unwrap();
            } else {
                println!("SORRY NOTHING");
            }
	}
        // Define the spacing between vertical lines (every 4th place).
        let line_spacing = 4;

        if bandwidth == "thick" {
            // Loop to draw the vertical lines.
            for bar_x in (0..75).step_by(line_spacing) {
                let rect = Rect::new(
                    (bar_x - 1) * zoom,
                    0, // Adjust this if you want the lines to start from a different Y coordinate.
                    1 * zoom as u32,  // Set the width of the line (1 pixel for vertical line).
                    16 * zoom as u32, // Set the height of the line (16 pixels high).
                );
                canvas.set_draw_color(_colors[0]);
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
            .map(|(left_sample, right_sample)| (((*left_sample + *right_sample) / 12.0)))
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
        for (i, &sample) in windowed_mixed_fft.iter().enumerate().take(4096) {
            mixed_f32[i] = sample as f32;
        }
        
        // compute the RFFT of the samples
        //let mut mixed_f32: [f32; 4096] = mixed.try_into().unwrap();
        let spectrum = microfft::real::rfft_4096(&mut mixed_f32);
        //println!("{}", spectrum.len());
        // since the real-valued coefficient at the Nyquist frequency is packed into the
        // imaginary part of the DC bin, it must be cleared before computing the amplitudes
        //spectrum[0].im = 0.0;

        let frequencies: Vec<f32> = (0..spectrum.len())
        .map(|i| i as f32 * 44100.0 / (2.0 * spectrum.len() as f32))
        .collect();
        // hard coding 44100 is a bad idea because windows setups can and will differ
        // does cpal return the samplerate?

        // Apply the weighting function
        apply_weighting(spectrum, &frequencies);

        // figure out linear interpolation
        let num_log_bins = 625*2; // Adjust this value as needed

        // Calculate the scaling factor for the logarithmic mapping
        let min_freq: f64 = 6.0; // Minimum frequency in Hz (adjust as needed)
        let max_freq: f64 = 22050.0; // Maximum frequency in Hz (adjust as needed)
        let log_min = min_freq.log2();
        let log_max = max_freq.log2();
        let log_bin_width = (log_max - log_min) / num_log_bins as f64;

        // Initialize a vector to store the logarithmic spectrum
        let mut log_spectrum = vec![0.0; num_log_bins];

        // Populate the logarithmic spectrum using linear interpolation
        for i in 0..num_log_bins {
            // Calculate the center frequency of the current bin in logarithmic scale
            let center_freq = min_freq * 2.0_f64.powf(log_bin_width * (i as f64 + 0.5) + log_min);
    
            // Find the two closest frequencies in the original spectrum
            let lower_freq_index = (center_freq * (spectrum.len() as f64 / max_freq)) as usize;
            let upper_freq_index = lower_freq_index + 1;
    
            // Ensure the indices are within bounds
            let lower_freq_index = lower_freq_index.min(spectrum.len() - 1);
            let upper_freq_index = upper_freq_index.min(spectrum.len() - 1);
    
            // Linearly interpolate between the two closest frequencies
            let lower_freq = frequencies[lower_freq_index];
            let upper_freq = frequencies[upper_freq_index];
    
            let lower_value = spectrum[lower_freq_index].l1_norm();
            let upper_value = spectrum[upper_freq_index].l1_norm();
    
            // Perform linear interpolation to estimate the value at the center frequency
            log_spectrum[i] = linear_interpolation(center_freq, lower_freq.into(), upper_freq.into(), lower_value.into(), upper_value.into());
        }

        // Convert the spectrum to amplitudes
        let amplitudes: Vec<_> = log_spectrum.iter().map(|c| c.l1_norm() as u8).collect();
        //println!("{amplitudes:?}");
        //assert_eq!(&amplitudes, &[0, 0, 0, 8, 0, 0, 0, 0]);

        // Convert the ring buffer to a regular Vec<u8> and send it through the channel
        match tx.send(ring_buffer.iter().copied().collect()) {
            Ok(_) => {
                // Send successful
            }
            Err(_err) => {
            }
        }
        
        match s.send(amplitudes.iter().copied().collect()) {
            Ok(_) => {
                // Send successful
            }
            Err(_err) => {
            }
        }
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

fn draw_window(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    _colors: &[Color],
    cgenex: &[Color],
    texture_creator: &TextureCreator<WindowContext>,
    font: &sdl2::ttf::Font,
    marlett: &sdl2::ttf::Font,
    oscstyle: &str,
    mode: u8,
    image_path: &str,
    is_button_clicked: bool,
    mx: i32,
    my: i32,
    peaks: Arc<Mutex<u8>>,
    specdraw: &str,
    bandwidth: &str,
) -> Result<(), String> {
    let mut visstatus: String = "".to_string();

    if mode == 0 {
        visstatus = "Spectrum Analyzer".to_string();
    } else if mode == 1 {
        visstatus = "Oscilloscope".to_string();
    } else if mode == 2 {
        visstatus = "Disabled".to_string();
    }
    let rect = Rect::new(0, 0, 606, 592);
    canvas.set_draw_color(cgenex[2]);
    canvas.fill_rect(rect).unwrap();

    canvas.set_draw_color(cgenex[5]);
    let rect = Rect::new(185, 28, 412, 556);
    canvas.draw_rect(rect).unwrap();

    listview_box(canvas, cgenex, 10, 8, 165, 545)?;

    //vis box
    draw_dropdown(canvas, cgenex, 206, 105, 362, 21)?;

    let tabtext: String = "Classic Visualization".to_string();
    let classivis: String = "Classic skins have a simple visualization in the main window. You can\nselect what kind of visualization here or click on the visualization to cycle\nthrough the modes.".to_string();
    let groupboxtext1: String = "Classic Visualization Settings".to_string();
    let groupboxtext2: String = "Spectrum Analyzer Options".to_string();
    let groupboxtext3: String = "Oscilloscope Options".to_string();
    let gbinfo1: String = "Coloring Style:".to_string();
    let gbinfo2: String = "Band line width:".to_string();
    let gbinfo3: String = "Oscilloscope drawing".to_string();
    let vis_text = &visstatus;

    tab(canvas, cgenex, 187, 8, &tabtext, font, texture_creator)?;

/*         //tab
        canvas.set_draw_color(cgenex[5]);
        let rect = Rect::new(187, 8, 116, 21);
        canvas.draw_rect(rect).unwrap();
        canvas.set_draw_color(cgenex[10]);
        let rect = Rect::new(188, 9, 114, 19);
        canvas.fill_rect(rect).unwrap();
        render_text(canvas, font, &tabtext, cgenex[1], 197, 11, texture_creator)?; */

    groupbox(canvas, &groupboxtext1, font, texture_creator, cgenex, 195, 37, 384, 125)?;
    groupbox(canvas, &groupboxtext2, font, texture_creator, cgenex, 195, 170, 384, 153)?;
    groupbox(canvas, &groupboxtext3, font, texture_creator, cgenex, 195, 333, 384, 46)?;

    render_text(canvas, font, vis_text, cgenex[1], 212, 108, texture_creator)?;
    render_text(canvas, font, &gbinfo1, cgenex[4], 206, 193, texture_creator)?;
    render_text(canvas, font, &gbinfo2, cgenex[4], 206, 216, texture_creator)?;
    render_text(canvas, font, &gbinfo3, cgenex[4], 206, 354, texture_creator)?;

    checkbox(canvas, cgenex, 206, 237, "Show Peaks", font, marlett, texture_creator, peaks, is_button_clicked, mx, my)?;

    slider_small(canvas, cgenex, 133, 44, 209, 264, "Falloff speed:", font, texture_creator, 5)?;
    slider_small(canvas, cgenex, 133, 44, 375, 264, "Peak falloff speed:", font, texture_creator, 5)?;

    button(canvas, cgenex, 10, 563, 165, 22, "Close", font, texture_creator, image_path, is_button_clicked, mx, my)?;

    radiobutton(canvas, cgenex, 297, 192, "Normal;Fire;Line", font, marlett, texture_creator, specdraw, is_button_clicked, mx, my)?;

    radiobutton(canvas, cgenex, 297, 216, "Thin;Thick", font, marlett, texture_creator, bandwidth, is_button_clicked, mx, my)?;

    radiobutton(canvas, cgenex, 350, 355, "Dots;Lines;Solid", font, marlett, texture_creator, oscstyle, is_button_clicked, mx, my)?;
    // Use the split_lines_and_create_textures function for classivis
    let tex2 = newline_handler(&classivis, font, texture_creator, cgenex[4])?;

    // Draw tex2
     let mut y = 56;
    for texture in tex2 {
        let texture_query = texture.query(); // Get the TextureQuery struct
        let w = texture_query.width;
        let h = texture_query.height;
        let target2 = Rect::new(206 as i32, y as i32, w, h);
        canvas.copy(&texture, None, Some(target2))?;
        y += h as i32;
    }
    Ok(())
}

fn main() -> Result<(), String> {
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
    
    // handle args
    let mut oscstyle = args.oscstyle.as_str(); // Convert String to &str
    let mut specdraw = args.specdraw.as_str();
    let zoom = args.zoom;
    let amp = args.amp;
    let mut mode = args.mode;
    let mut bandwidth = args.bandwidth.as_str();
    let mut peakfo = args.peakfo;
    let mut barfo = args.barfo;
    let mut peaks = Arc::new(Mutex::new(args.peaks));

    if args.peakfo <= 1 {
        peakfo = 1;
    } else if args.peakfo >= 5 {
        peakfo = 5;
    }

    if args.barfo <= 1 {
        barfo = 1;
    } else if args.barfo >= 5 {
        barfo = 5;
    }

    let mut bars = [Bar {
        height: 0.0,
        height2: 0.0,
        peak: 0.0,
        gravity: 0.0,
        bargrav: barfo as f64 / 3.0,
    }; NUM_BARS];

    let mut mouse_x: i32 = 0;
    let mut mouse_y: i32 = 0;

    // set up sdl2
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;
    //sdl2::hint::set("SDL_HINT_RENDER_SCALE_QUALITY", "0");
    let window = video_subsystem
        .window("Winamp Mini Visualizer (in Rust)", (WINDOW_WIDTH * zoom) as u32, (WINDOW_HEIGHT * zoom) as u32)
        .position_centered()
        .build()
        .unwrap();

    let window2 = video_subsystem
        .window("RustampVis Preferences", 606 as u32, 592 as u32)
        .position_centered()
        .build()
        .unwrap();
    let mut canvas = window.into_canvas().build().unwrap();

    let mut canvas2 = window2.into_canvas().build().unwrap();
    let ttf_context = sdl2::ttf::init().unwrap();
    let texture_creator = canvas2.texture_creator();
    let mut font = ttf_context.load_font(&"font/tahoma.ttf", 11)?;
    let mut vectorgfx = ttf_context.load_font(&"font/marlett.ttf", 15)?;
    font.set_hinting(sdl2::ttf::Hinting::Mono);
    vectorgfx.set_hinting(sdl2::ttf::Hinting::Mono);

    let mut event_pump = sdl_context.event_pump().unwrap();

    let surface =
    sdl2::surface::Surface::from_file(&"NORMAL.PNG").map_err(|err| format!("failed to load cursor image: {}", err))?;
    let cursor = Cursor::from_surface(surface, 0, 0)
        .map_err(|err| format!("failed to load cursor: {}", err))?;
    cursor.set();

    // Load the custom viscolor.txt file
    let mut viscolors = viscolors::load_colors(&args.viscolor);
    // extract relevant osc colors from the array
    let mut osc_colors = osccolors(&viscolors);
    let mut peakrgb = peakc(&viscolors);
    
    // The vector to store captured audio samples.
    let audio_data = Arc::new(Mutex::new(Vec::<u8>::new()));
    let spec_data = Arc::new(Mutex::new(Vec::<u8>::new()));

    // Create a blocking receiver to get audio samples from the audio stream loop.
    let (tx, rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = unbounded();
    let (s, r): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = unbounded();

    // Start the audio stream loop in a separate thread.
    thread::spawn(move || audio_stream_loop(tx, s, Some(selected_device_index), amp));
    //thread::spawn(move || fltk());

    let image_path = "gen_ex.png";
    let genex_colors = genex(image_path);
    let mut is_button_clicked = false;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running
                },
                Event::Window { win_event: WindowEvent::Close, .. } => {
                    // SDL_GetWindowByID, SDL_HideWindow it
                },
                Event::KeyDown { window_id: 1, keycode: Some(Keycode::R), .. } => {
                    // Reload the viscolors data when "R" key is pressed
                    let new_viscolors = viscolors::load_colors(&args.viscolor);
                    let new_osc_colors = osccolors(&new_viscolors);
                    let new_peakrgb = peakc(&new_viscolors);
                    viscolors = new_viscolors;
                    osc_colors = new_osc_colors;
                    peakrgb = new_peakrgb;
                }
                Event::KeyDown { window_id: 1, keycode: Some(Keycode::B), .. } => {
                    // switch bandwidth
                    if bandwidth == "thick"{
                        switch_bandwidth(&mut bandwidth);
                    } else if bandwidth == "thin" {
                        switch_bandwidth(&mut bandwidth);
                    }
                }
                Event::MouseButtonDown { window_id: 1, mouse_btn: MouseButton::Right, .. } => {
                    if mode == 1{
                        switch_oscstyle(&mut oscstyle);
                    } else if mode == 0 {
                        switch_specstyle(&mut specdraw);
                    }
                }
                Event::MouseButtonDown { window_id: 1, mouse_btn: MouseButton::Left, .. } => {
                    mode = (mode + 1) % 3;
                    //println!("{mode}")
                }
                Event::MouseMotion { window_id: 2, x, y, .. } => {
                    // Handle mouse motion events
                    println!("Mouse moved to ({}, {})", x, y);
                }
                Event::MouseButtonDown { window_id: 2, mouse_btn, x, y, .. } => {
                    // Handle mouse button down events
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        is_button_clicked = true;
                        mouse_x = x;
                        mouse_y = y;
                    }
                }
                Event::MouseButtonUp { window_id: 2, mouse_btn, x, y, .. } => {
                    // Handle mouse button down events
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        is_button_clicked = false;
                        mouse_x = x;
                        mouse_y = y;
                    }
                }
                _ => {}
            }
        }

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

        //println!("{}", sdl2::get_framerate());
        draw_visualizer(&mut canvas, &viscolors, &osc_colors, peakrgb, &*audio_data, &*spec_data, oscstyle, specdraw, mode, &bandwidth, zoom, &mut bars, peakfo, peaks.clone()/* , modern*/);
        draw_window(&mut canvas2, &viscolors, &genex_colors, &texture_creator, &font, &vectorgfx, oscstyle, mode, image_path, is_button_clicked, mouse_x, mouse_y, peaks.clone(), specdraw, &bandwidth)?;

        // draw the cool shit
        canvas.present();
        canvas2.present();

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

fn osccolors(colors: &[Color]) -> Vec<Color> {
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

fn peakc(colors: &[Color]) -> Color {
    if colors.len() == 35 {
        colors[34]
    } else {
        colors[23]
    }
}

fn genex(image_path: &str) -> Vec<Color> {
    // Open an image file
    let img = image::open(image_path).unwrap();

    // Initialize the vector to hold the colors
    let mut color_vector: Vec<Color> = Vec::new();

    // Iterate over the image pixels and convert them to SDL2 Color
    for y in 0..1 {
        for x in (48..96).step_by(2) {
            let pixel = img.get_pixel(x, y);
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];
            let color = Color::RGB(r, g, b);
            color_vector.push(color);
            //println!("{:?}", pixel);
        }
    }

    color_vector
}
