//ChatGPT assisted me with this.
//Base code came from https://stackoverflow.com/questions/72181090/getting-pixels-of-many-different-colors-when-drawing-with-sdl-in-rust

extern crate sdl2;

use clap::Parser;

use sdl2::image::{InitFlag, LoadSurface};
use sdl2::keyboard::Keycode;
use sdl2::mouse::Cursor;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::rect::Point;
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
use std::fs;

use image::GenericImageView;

mod viscolors;
mod comctl;

use comctl::render_individual_letters;
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
const NUM_BARS: usize = WINDOW_WIDTH as usize;

const SCROLL_SPEED: f64 = 0.75;
static mut SHIFT_AMOUNT: usize = 0;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {

    /// Name of the custom viscolor.txt file, supports images in the viscolor.txt format as well.
    #[arg(short, long, default_value = "viscolor.txt")]
    viscolor: String,
    
    /// Index of the audio device to use
    #[arg(short, long)]
    device: Option<usize>,

    /// Zoom factor
    #[arg(short, long, default_value = "7")]
    zoom: i32,


    /* /// Modern Skin style visualization
    #[arg(short, long, default_value = "0")]
    modern: bool,*/

    /// Name of the config file.
    #[arg(short, long, default_value = "rustampvis.ini")]
    configini: String,

    /// Debug
    #[arg(long)]
    debug: bool,
}

#[derive(Debug)]
struct DeviceSelection {
    index: usize,
    name: String,
}

#[derive(Debug)]
struct WinampConfig {
    sa: Option<i32>,
    safire: Option<i32>,
    sa_peaks: Option<i32>,
    safalloff: Option<i32>,
    sa_peak_falloff: Option<i32>,
    sa_amp: Option<i32>,
    vu_peaks: Option<i32>,
    vu_style: Option<i32>,
    vu_peak_fall_off: Option<i32>,
    prefs_tab: Option<i32>,
    aot: Option<i32>,
    vu_rms: Option<i32>,
    non_visualizer_sections: String, // Store non-visualizer sections as a string
    // Add more fields as needed
}

fn winamp_ini(content: &str) -> WinampConfig {
    let mut winamp_config = WinampConfig {
        sa: None,
        safire: None,
        sa_peaks: None,
        safalloff: None,
        sa_peak_falloff: None,
        sa_amp: None,
        vu_peaks: None,
        vu_style: None,
        vu_peak_fall_off: None,
        prefs_tab: None,
        aot: None,
        vu_rms: None,
        non_visualizer_sections: String::new(),
    };

    let mut in_winamp_section = false;

    for line in content.lines() {
        let trimmed_line = line.trim();

        if trimmed_line == "[Winamp]" || trimmed_line == "[RustampVis]" {
            in_winamp_section = true;
        } else if in_winamp_section {
            if let Some(key_value) = trimmed_line.split_once('=') {
                let key = key_value.0.trim();
                let value = key_value.1.trim();

                match key {
                    "sa" => winamp_config.sa = value.parse().ok(),
                    "safire" => winamp_config.safire = value.parse().ok(),
                    "sa_peaks" => winamp_config.sa_peaks = value.parse().ok(),              
                    "safalloff" => winamp_config.safalloff = value.parse().ok(),
                    "sa_amp" => winamp_config.sa_amp = value.parse().ok(),
                    "sa_peak_falloff" => winamp_config.sa_peak_falloff = value.parse().ok(),
                    "vu_peaks" => winamp_config.vu_peaks = value.parse().ok(),
                    "vu_style" => winamp_config.vu_style = value.parse().ok(),
                    "vu_peak_fall_off" => winamp_config.vu_peak_fall_off = value.parse().ok(),
                    "prefs_tab" => winamp_config.prefs_tab = value.parse().ok(),
                    "aot" => winamp_config.aot = value.parse().ok(),
                    "vu_rms" => winamp_config.vu_rms = value.parse().ok(),
                    _ => {}
                }
            }
        } else {
            // Store non-visualizer sections
            winamp_config.non_visualizer_sections.push_str(&format!("{}\n", trimmed_line));
        }
    }

    winamp_config
}

#[derive(Debug)]
#[derive(Copy)]
#[derive(Clone)]
struct Bar {
    height: i32,
    height2: f64,
    peak: f64,
    gravity: f64,
    bargrav: f64,
    vumeter: i32,
    vumeterpeak: f32,
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

fn switch_oscstyle(oscstyle: &mut String) {
    match oscstyle.as_str() { // Convert the String to &str for matching
        "dots" => *oscstyle = "lines".to_string(),
        "lines" => *oscstyle = "solid".to_string(),
        "solid" => *oscstyle = "dots".to_string(),
        _ => println!("Invalid oscilloscope style. Supported styles: dots, lines, solid."),
    }
}

fn switch_specstyle(specdraw: &mut String) {
    match specdraw.as_str() { // Convert the String to &str for matching
        "normal" => *specdraw = "fire".to_string(),
        "fire" => *specdraw = "line".to_string(),
        "line" => *specdraw = "normal".to_string(),
        _ => println!("Invalid analyzer style. Supported styles: normal, fire, line."),
    }
}

fn switch_bandwidth(bandwidth: &mut String) {
    match bandwidth.as_str() { // Convert the String to &str for matching
        "thick" => *bandwidth = "thin".to_string(),
        "thin" => *bandwidth = "thick".to_string(),
        _ => println!("Invalid bandwidth. Supported bandwidths: thick, thin."),
    }
}

fn switch_vustyle(vudraw: &mut String) {
    match vudraw.as_str() { // Convert the String to &str for matching
        "normal" => *vudraw = "fire".to_string(),
        "fire" => *vudraw = "line".to_string(),
        "line" => *vudraw = "normal".to_string(),
        _ => println!("Invalid analyzer style. Supported styles: normal, fire, line."),
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

fn process_fft_data(bars: &mut [Bar], fft_iter: &mut std::slice::Iter<f64>, bandwidth: &str, peakfo: u8, barfo: u8, debug: bool, mut debug_vector: Vec<f64>,) {
    //print!("{}", barfo as f64 / 2.75);

    let bv_vec: Vec<f64> = vec![0.19, 0.422, 0.75, 1.0, 2.0];
    let bv_index = (barfo as usize).saturating_sub(1).min(bv_vec.len() - 1);
    let bv_value = bv_vec[bv_index] as f64;

    let pv_vec: Vec<f64> = vec![1.0, 2.0, 6.0, 14.0, 20.0]; // peaks dont fall like they do in winamp/WACUP
    // now they kinda do
    let pv_index = (peakfo as usize).saturating_sub(1).min(pv_vec.len() - 1);
    let pv_value = pv_vec[pv_index] as f64;
    let threshold: f64 = ((pv_value as f64)) / 14.0 + 0.75;
    //println!{"{}, {}", pv_value, threshold};

    let bvalue: f64 = bv_value;
    let pvalue: f64 = pv_value / 64.0;

    if !debug {
        if bandwidth == "thick" {
            for bars_chunk in bars.chunks_mut(4) {
                let mut sum = 0.0;

                for _ in 0..24 * 2 {
                    if let Some(fft_value) = fft_iter.next() {
                        sum += *fft_value as f64;
                    } else {
                        break;
                    }
                }

                for bar in bars_chunk.iter_mut().take(4) {
                    bar.height = sum as i32 / (25.0 * 2.0) as i32;
                    if bar.height >= 15 {
                        bar.height = 15;
                    }
                }
            }
        } else {
            for bars_chunk in bars.chunks_mut(1) {
                let mut sum = 0.0;

                for _ in 0..6 * 2 {
                    if let Some(fft_value) = fft_iter.next() {
                        sum += *fft_value as f64;
                    } else {
                        break;
                    }
                }

                for bar in bars_chunk.iter_mut() {
                    bar.height = sum as i32 / (7.0 * 2.0) as i32;
                    if bar.height >= 15 {
                        bar.height = 15;
                    }
                }
            }
        }
    }

    if debug {
        // Shift the debug vector to the right on every tick
        shift_vector_to_right(&mut debug_vector);

        // Set the height of bars based on the shifted debug vector
        for (bar, &debug_value) in bars.iter_mut().zip(debug_vector.iter()) {
            bar.height = debug_value as i32;
        }
    }

    for i in 0..NUM_BARS {
        bars[i].height2 -= bars[i].bargrav + bvalue;

        if bars[i].height2 <= bars[i].height as f64 {
            bars[i].height2 = bars[i].height as f64;
        if debug{
            if bars[i].height2 > bars[i].peak {
                bars[i].gravity = 0.0;
                bars[i].peak = bars[i].height2;
            } else {
                if bars[i].gravity <= 16.0 {
                    bars[i].gravity += pvalue;
                }
                bars[i].peak = f64::max(0.0, bars[i].peak - f64::max(0.0, bars[i].gravity - threshold));
                // Check if height2 is within the specified range (14.0 to 14.99)
                if (bars[i].height2 >= 14.0) && (bars[i].height2 <= 14.99) || (bv_index == 4) && (bars[i].height2 >= 13.0) && (bars[i].height2 <= 14.99) {
                    // Offset peak by -1
                    bars[i].peak -= 0.25;
                }
                if bars[i].peak == 0.0 {
                    bars[i].peak = -3.0;
                }
            }

        }

        }
        if !debug {
            if bars[i].height2 > bars[i].peak {
                bars[i].gravity = 0.0;
                bars[i].peak = bars[i].height2;
            } else {
                if bars[i].gravity <= 16.0 {
                    bars[i].gravity += pvalue;
                }
                bars[i].peak = f64::max(0.0, bars[i].peak - f64::max(0.0, bars[i].gravity - threshold));
                if bars[i].peak < bars[i].height2 {
                    bars[i].peak = bars[i].height2;
                }
            }
            // check if height2 is within the range (14.0 to 14.99)
            // also check if height 2 is within 13 and 14.99 and
            // if our bar fall off is set to 4, otherwise chaos will ensue
            if (bars[i].height2 >= 14.0) && (bars[i].height2 <= 14.99) || (bv_index == 4) && (bars[i].height2 >= 13.0) && (bars[i].height2 <= 14.99) {
                // set peak to 14.0 after bars hit value 15
                bars[i].peak = 14.0;
            }
            if bars[i].peak == 0.0 {
                bars[i].peak = -3.0;
            }
            //println!("{}, {}", bars[i].peak, i);
            //println!("{}, {}", bars[i].height2, i);
        }
    }
}

// Define shift_vector_to_right function
fn shift_vector_to_right(vector: &mut Vec<f64>) {
    static mut SHIFT_AMOUNT: isize = 0;
    static mut DIRECTION: isize = 1;

    unsafe {
        SHIFT_AMOUNT += DIRECTION;

        if SHIFT_AMOUNT == vector.len() as isize -1 {
            // Reached the end, change direction to negative
            DIRECTION = -1;
        } else if SHIFT_AMOUNT == 0 {
            // Reached the beginning, change direction to positive
            DIRECTION = 1;
        }

        vector.rotate_right(SHIFT_AMOUNT.abs() as usize);
        //println!("{}", SHIFT_AMOUNT);
    }
}

fn draw_visualizer(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    _colors: &[Color],
    osc_colors: &[Color],
    vu_color: &[Color],
    peak_color: Color,
    ys: &[f32],
    ys1: &[f32],
    ys2: &[f32],
    fft: &[f32],
    oscstyle: &str,
    specdraw: &str,
    vudraw: &str,
    mode: u8,
    bandwidth: &str,
    zoom: i32,
    bars: &mut [Bar],
    peakfo: u8,
    vu_peak_fall_off: u8,
    barfo: u8,
    vu_rms: u8,
    peaks: Arc<Mutex<u8>>,
    vu_peaks: Arc<Mutex<u8>>,
    amp: u8,
    debug: bool,
    debug_vector: Vec<f64>,
) {
    let peaks_unlocked = peaks.lock().unwrap();
    let vu_peaks_unlocked = vu_peaks.lock().unwrap();
    let xs: Vec<i32> = (0..75).collect();
    let ys: Vec<f32> = ys.iter().step_by(8).map(|&sample| ((sample * amp as f32)) as f32/* * WINDOW_HEIGHT / 16*/).collect();
    let ys1: Vec<f32> = ys1.iter().map(|&sample| ((sample * amp as f32)) as f32/* * WINDOW_HEIGHT / 16*/).collect();
    let ys2: Vec<f32> = ys2.iter().map(|&sample| ((sample * amp as f32)) as f32/* * WINDOW_HEIGHT / 16*/).collect();   // cast to i32
    let fft: Vec<f64> = fft.iter()
    .map(|&sample| ((sample * amp as f32 / 8.0)) as f64)
    .collect(); // cast to i32

    let mut last_y = 0;
    let mut top: i32;
    let mut bottom: i32;
    let mut top2: i32;
    let mut peak1: i32;
    let mut peak2: i32;

    // analyzer stuff
    process_fft_data(bars, &mut fft.iter(), bandwidth, peakfo, barfo, debug, debug_vector);
    process_audio_data(bars, &mut ys1.iter(), &mut ys2.iter(), vu_peak_fall_off, vu_rms);

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
    if mode == 2{
        for (x, y) in xs.iter().zip(ys.iter()) {
            let x = *x;
            let y = *y;

            let y: i32 = ((y / 8.0) + 7.6) as i32;
    
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

    } else if mode == 1{
        for (i, bar) in bars.iter().enumerate() {
            let x = i as i32 * zoom as i32;
            let y = -bar.height2 as i32 + 15;
            
                top = y + 1;
                bottom = 15;

            for dy in top..=bottom {
                let color_index: usize;
                if specdraw == "normal"{
                    color_index = (dy as usize).wrapping_add(2) % _colors.len();
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
            let bar_height = -bar.peak + 15.0;
            let peaki32: i32 = bar_height as i32;

            let rect = Rect::new(
                bar_x, peaki32.wrapping_mul(zoom as i32),
                zoom as u32,
                zoom as u32,
            );
            let color = peak_color;
            canvas.set_draw_color(color);
            if *peaks_unlocked == 1 {
                canvas.fill_rect(rect).unwrap();
            } else {
                //println!("SORRY NOTHING");
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
    } else if mode == 3{

        for i in 0..75{
            let num = bars[i].vumeter;
            //println!("{i}, {num}");
        }

        top = -bars[0].vumeter as i32;
        top2 = -bars[1].vumeter as i32;
        bottom = 75;

        peak1 = -bars[0].vumeterpeak as i32 - 1;
        peak2 = -bars[1].vumeterpeak as i32 - 1;

        for dy in top..=bottom {
            let mut color_index: usize;
                if vudraw == "fire" {
                    color_index = ((-dy + top - 17) as usize) % vu_color.len();
                } else if vudraw == "normal" {
                    color_index = (-dy as usize) % vu_color.len();
                } else if vudraw == "line" {
                    color_index = (-top as usize) % vu_color.len();
                } else {
                    color_index = 0;
                }
            let bar_color = vu_color[color_index];
            let rect = Rect::new((-dy) * zoom, 1 * zoom, zoom as u32, 7 * zoom as u32);
            //let rect2 = Rect::new((-dy) * zoom, 9 * zoom, zoom as u32, 7 * zoom as u32);
            canvas.set_draw_color(bar_color);
            canvas.fill_rect(rect).unwrap();
        }

        for dy in top2..=bottom {
            let mut color_index: usize;
                if vudraw == "fire" {
                    color_index = ((-dy + top2 - 17) as usize) % vu_color.len();
                } else if vudraw == "normal" {
                    color_index = (-dy as usize) % vu_color.len();
                } else if vudraw == "line" {
                    color_index = (-top2 as usize) % vu_color.len();
                } else {
                    color_index = 0;
                }
            let bar_color = vu_color[color_index];
            let rect = Rect::new((-dy) * zoom, 9 * zoom, zoom as u32, 7 * zoom as u32);
            canvas.set_draw_color(bar_color);
            canvas.fill_rect(rect).unwrap();
        }

        for dy in peak1..=bottom {
            let color = peak_color;
            let rect = Rect::new((-peak1) * zoom, 1 * zoom, zoom as u32, 7 * zoom as u32);
            canvas.set_draw_color(color);
            if *vu_peaks_unlocked == 1 {
                canvas.fill_rect(rect).unwrap();
            } else {
                //println!("SORRY NOTHING");
            }
        }

        for dy in peak2..=bottom {
            let color = peak_color;
            let rect = Rect::new((-peak2) * zoom, 9 * zoom, zoom as u32, 7 * zoom as u32);
            canvas.set_draw_color(color);
            if *vu_peaks_unlocked == 1 {
                canvas.fill_rect(rect).unwrap();
            } else {
                //println!("SORRY NOTHING");
            }
        }
    } else if mode == 0{
    }
}

fn calculate_rms(samples: &[f32], take: i32) -> f32 {
    // Take the specified number of samples or use all if `take` is greater than the length
    let samples_to_process = samples.iter().take(take.min(samples.len().try_into().unwrap()).try_into().unwrap());

    // Calculate the sum of squared samples
    let sum_of_squares: f32 = samples_to_process.clone().map(|&x| x * x).sum();

    // Calculate the mean square value
    let mean_square = sum_of_squares / samples_to_process.len() as f32;

    // Calculate the root mean square (RMS)
    let rms = mean_square.sqrt();

    rms
}

fn process_audio_data(bars: &mut [Bar], ys: &mut std::slice::Iter<f32>, ys2: &mut std::slice::Iter<f32>, vu_peak_fall_off: u8, rms_v: u8) {

    let ta_vec: Vec<i32> = vec![16384, 8192, 4096, 2048, 1024];
    let ta_index = (rms_v as usize).saturating_sub(1).min(ta_vec.len() - 1);
    let ta_value = ta_vec[ta_index] as i32;

    // Calculate RMS values for the first 75 samples for ys and ys2
    let rms_ys = calculate_rms(&ys.map(|&x| x).collect::<Vec<_>>(), ta_value);
    let rms_ys2 = calculate_rms(&ys2.map(|&x| x).collect::<Vec<_>>(), ta_value);

    println!("{ta_value}");
    // Update the second entry of vumeter field
    bars[0].vumeter = rms_ys as i32 - 1;
    if bars[0].vumeter >= 74 {
        bars[0].vumeter = 74;
    }
    bars[1].vumeter = rms_ys2 as i32 - 1;
    if bars[1].vumeter >= 74 {
        bars[1].vumeter = 74;
    }

    for i in 0..NUM_BARS {
        bars[i].vumeterpeak -= vu_peak_fall_off as f32 / 1.25;

        if bars[i].vumeterpeak <= bars[i].vumeter as f32 {
            bars[i].vumeterpeak = bars[i].vumeter as f32;
        }

        if bars[i].vumeterpeak >= 73.0 {
            bars[i].vumeterpeak = 73.0;
        }

        if bars[i].vumeterpeak == -1.0 {
            bars[i].vumeterpeak = -3.0;
        }
    }

    bars[2].vumeter = bars[0].vumeter + bars[1].vumeter / 2.0 as i32;
}

fn audio_stream_loop(tx: Sender<Vec<f32>>, tx_l: Sender<Vec<f32>>, tx_r: Sender<Vec<f32>>, s: Sender<Vec<f32>>, selected_device_index: Option<usize>) {
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
    let mut ring_buffer: VecDeque<f32> = VecDeque::with_capacity(593); //ironically the WASAPI buffer is apparently 2048 anyway...?

    let mut ring_buffer_left: VecDeque<f32> = VecDeque::with_capacity(16384);
    let mut ring_buffer_right: VecDeque<f32> = VecDeque::with_capacity(16384);

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Convert f32 samples and collect them into a Vec<f32>
    let left: Vec<f32> = data
        .iter()
        .step_by(2)
        .map(|&sample| (((-sample)) * 127.5))
        .collect();

    let right: Vec<f32> = data
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&sample| (((-sample)) * 127.5))
        .collect();

    let left_fft: Vec<f32> = data
        .iter()
        .map(|&sample| (((-sample)) * 127.5))
        .collect();

    let right_fft: Vec<f32> = data
        .iter()
        .skip(1)
        .map(|&sample| (((-sample)) * 127.5))
        .collect();
        
    let mixed: Vec<f32> = left
        .iter()
        .zip(right.iter())
        .map(|(left_sample, right_sample)| (((*left_sample + *right_sample) / 2.0)))
        .collect();

    let mixed_fft: Vec<f32> = left_fft
        .iter()
        .zip(right_fft.iter())
        .map(|(left_sample_fft, right_sample_fft)| (*left_sample_fft + *right_sample_fft) / 12.0)
        .collect();
    
        // Extend the ring buffer with the new samples
        for mixed in &mixed {
            if ring_buffer.len() == ring_buffer.capacity() {
                ring_buffer.pop_front(); // Remove the oldest sample when the buffer is full
            }
            ring_buffer.push_back(*mixed);
        }

        for left_sample in &left {
            if ring_buffer_left.len() == ring_buffer_left.capacity() {
                ring_buffer_left.pop_back(); // Remove the oldest sample when the buffer is full
            }
            ring_buffer_left.push_front(*left_sample);
        }

        for right_sample in &right {
            if ring_buffer_right.len() == ring_buffer_right.capacity() {
                ring_buffer_right.pop_back(); // Remove the oldest sample when the buffer is full
            }
            ring_buffer_right.push_front(*right_sample);
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
        let amplitudes: Vec<_> = log_spectrum.iter().map(|c| c.l1_norm() as f32).collect();
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

        match tx_l.send(ring_buffer_left.iter().copied().collect()) {
            Ok(_) => {
                // Send successful
            }
            Err(_err) => {
            }
        }

        match tx_r.send(ring_buffer_right.iter().copied().collect()) {
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
    cgenex: &[Color],
    texture_creator: &TextureCreator<WindowContext>,
    tahoma: &sdl2::ttf::Font,
    tahoma_bold: &sdl2::ttf::Font,
    marlett: &sdl2::ttf::Font,
    oscstyle: Arc<Mutex<String>>,
    mode: u8,
    image_path: &str,
    is_button_clicked: bool,
    mx: &mut i32,
    my: &mut i32,
    peaks: Arc<Mutex<u8>>,
    vu_peaks: Arc<Mutex<u8>>,
    specdraw: Arc<Mutex<String>>,
    vu_style: Arc<Mutex<String>>,
    vu_style_num: Arc<Mutex<i32>>,
    bandwidth: Arc<Mutex<String>>,
    mut peakfo: &mut u8,
    mut vupeakfo: &mut u8,
    mut barfo: &mut u8,
    mut vu_rms: &mut u8,
    mut amp: &mut u8,
    scroll: f64,
    ys: &mut [Bar],
    mut config_content: String,
    winamp_config: &mut WinampConfig,
    config_file_path: String,
    mut safire: i32,
    device_in_use: &String,
    mut prefs_id: &mut i32,
    aot: Arc<Mutex<u8>>,
) -> Result<(), String> {
    let mut visstatus: String = "".to_string();
    let amp_str: String = amp.to_string();
/*     let specdraw_mutex = Arc::new(Mutex::new(specdraw.to_string()));
    let oscstyle_mutex = Arc::new(Mutex::new(oscstyle.to_string()));
    let bandwidth_mutex = bandwidth.lock().unwrap(); */
    let mut vu_style_mutex = vu_style_num.lock().unwrap();
    let br: f64 = (ys[2].vumeter + 1) as f64 * 2.27;
    //println!("{ys:?}");

    //println!("{}", br as u8);

    if mode == 0 {
        visstatus = "Disabled".to_string();
    } else if mode == 1 {
        visstatus = "Spectrum Analyzer".to_string();
    } else if mode == 2 {
        visstatus = "Oscilloscope".to_string();
    } else if mode == 3 {
        visstatus = "VU Meter".to_string();
    }

    let rect = Rect::new(0, 0, 606, 592);
    canvas.set_draw_color(cgenex[2]);
    canvas.fill_rect(rect).unwrap();

    canvas.set_draw_color(cgenex[5]);
    let rect = Rect::new(185, 28, 412, 556);
    canvas.draw_rect(rect).unwrap();

    listview_box(canvas, cgenex, 10, 8, 165, 545)?;

    //vis box

    let tabtext: String = "Classic Visualization;Additional Options".to_string();
    let classivis: String = "Classic skins have a simple visualization in the main window. You can\nselect what kind of visualization here or click on the visualization to cycle\nthrough the modes.".to_string();
    let groupboxtext1: String = "Classic Visualization Settings".to_string();
    let groupboxtext2: String = "Spectrum Analyzer Options".to_string();
    let groupboxtext3: String = "Oscilloscope Options".to_string();
    let gbinfo1: String = "Coloring Style:".to_string();
    let gbinfo2: String = "Band line width:".to_string();
    let gbinfo3: String = "Oscilloscope drawing".to_string();
    let vis_text = &visstatus;

    tab(canvas, cgenex, 187, 8, &tabtext, &mut prefs_id, tahoma, texture_creator, is_button_clicked, mx, my)?;
    button(canvas, cgenex, 10, 563, 165, 22, "Close", tahoma, texture_creator, image_path, is_button_clicked, *mx, *my)?;
    render_individual_letters(canvas, tahoma, "RustampVis", cgenex[4], 10, 464, scroll + ys[0].height2 * 2.0, 5, texture_creator, sdl2::rect::Rect::new(164, 88, 164, 88), br)?;

    render_text(canvas, tahoma_bold, "Skins", cgenex[4], 42, 12, texture_creator)?;
    render_text(canvas, tahoma, "Classic Skins", cgenex[4], 62, 28, texture_creator)?;
    
    if *prefs_id == 0 {

        groupbox(canvas, &groupboxtext1, tahoma, texture_creator, cgenex, 195, 37, 384, 125)?;
        groupbox(canvas, &groupboxtext2, tahoma, texture_creator, cgenex, 195, 170, 384, 153)?;
        groupbox(canvas, &groupboxtext3, tahoma, texture_creator, cgenex, 195, 333, 384, 46)?;
        groupbox(canvas, "VU Meter Options", tahoma, texture_creator, cgenex, 195, 388, 384, 125)?;
        checkbox(canvas, cgenex, 206, 435, "Show Peaks", tahoma, marlett, texture_creator, vu_peaks.clone(), is_button_clicked, mx, my)?;
        render_text(canvas, tahoma, "Peak falloff speed:", cgenex[4], 210, 458, texture_creator)?;
        slider_small(canvas, cgenex, 133, 44, 210, 485, texture_creator, &mut vupeakfo, 5, image_path, is_button_clicked, mx, my)?;
        render_text(canvas, tahoma, "RMS intensity:", cgenex[4], 380, 458, texture_creator)?;
        slider_small(canvas, cgenex, 133, 44, 380, 485, texture_creator, &mut vu_rms, 5, image_path, is_button_clicked, mx, my)?;
        groupbox(canvas, "Device in use:", tahoma, texture_creator, cgenex, 195, 520, 384, 46)?;
        render_text(canvas, tahoma, device_in_use, cgenex[4], 209, 540, texture_creator)?;
        render_text(canvas, tahoma, &gbinfo1, cgenex[4], 206, 409, texture_creator)?;
        radiobutton(canvas, cgenex, 297, 410, "Normal;Fire;Line", tahoma, marlett, texture_creator, vu_style.clone(), is_button_clicked, *mx, *my)?;

        // vis box
        draw_dropdown(canvas, cgenex, 206, 105, 362, 21)?;
        render_text(canvas, tahoma, vis_text, cgenex[1], 212, 108, texture_creator)?;
        render_text(canvas, tahoma, &gbinfo1, cgenex[4], 206, 192, texture_creator)?;
        render_text(canvas, tahoma, &gbinfo2, cgenex[4], 206, 216, texture_creator)?;
        render_text(canvas, tahoma, &gbinfo3, cgenex[4], 206, 354, texture_creator)?;
    
        checkbox(canvas, cgenex, 206, 237, "Show Peaks", tahoma, marlett, texture_creator, peaks.clone(), is_button_clicked, mx, my)?;
    
        render_text(canvas, tahoma, "Falloff speed:", cgenex[4], 209, 264, texture_creator)?;
        slider_small(canvas, cgenex, 133, 44, 209, 289, texture_creator, &mut barfo, 5, image_path, is_button_clicked, mx, my)?;
        render_text(canvas, tahoma, "Peak falloff speed:", cgenex[4], 375, 264, texture_creator)?;
        slider_small(canvas, cgenex, 133, 44, 375, 289, texture_creator, &mut peakfo, 5, image_path, is_button_clicked, mx, my)?;
    
        render_text(canvas, tahoma, "Gain:", cgenex[4], 209, 137, texture_creator)?;
        slider_small(canvas, cgenex, 260, 44, 280, 140, texture_creator, &mut amp, 15, image_path, is_button_clicked, mx, my)?;
    
        render_text(canvas, tahoma, &amp_str, cgenex[4], 552, 137, texture_creator)?;
    
        radiobutton(canvas, cgenex, 297, 192, "Normal;Fire;Line", tahoma, marlett, texture_creator, specdraw.clone(), is_button_clicked, *mx, *my)?;
    
        radiobutton(canvas, cgenex, 297, 216, "Thin;Thick", tahoma, marlett, texture_creator, bandwidth.clone(), is_button_clicked, *mx, *my)?;
    
        radiobutton(canvas, cgenex, 350, 355, "Dots;Lines;Solid", tahoma, marlett, texture_creator, oscstyle.clone(), is_button_clicked, *mx, *my)?;
        // Use the split_lines_and_create_textures function for classivis
        let tex2 = newline_handler(&classivis, tahoma, texture_creator, cgenex[4])?;
    
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
    }

    if *prefs_id == 1 {
        checkbox(canvas, cgenex, 206, 50, "Always-on-Top", tahoma, marlett, texture_creator, aot.clone(), is_button_clicked, mx, my)?;
    }


    let safalloff_str = format!("safalloff={}", winamp_config.safalloff.unwrap_or_default());
    if config_content.contains(&safalloff_str) {
        config_content = config_content.replace(&safalloff_str, &format!("safalloff={}", *barfo - 1));
        //println!("safalloff: {:?}, barfo: {}", winamp_config.safalloff, *barfo);
    }

    let saamp_str = format!("sa_amp={}", winamp_config.sa_amp.unwrap_or_default());
    if config_content.contains(&saamp_str) {
        config_content = config_content.replace(&saamp_str, &format!("sa_amp={}", *amp));
        //println!("safalloff: {:?}, barfo: {}", winamp_config.safalloff, *barfo);
    }

    let sa_peak_falloff_str = format!("sa_peak_falloff={}", winamp_config.sa_peak_falloff.unwrap_or_default());
    if config_content.contains(&sa_peak_falloff_str) {
        config_content = config_content.replace(&sa_peak_falloff_str, &format!("sa_peak_falloff={}", *peakfo - 1));
        //println!("sa_peak_falloff: {:?}, peakfo: {}", winamp_config.sa_peak_falloff, *peakfo);
    }

    let sa_str = format!("sa={}", winamp_config.sa.unwrap_or_default());
    if config_content.contains(&sa_str) {
        config_content = config_content.replace(&sa_str, &format!("sa={}", mode));
        //println!("sa: {:?}, mode: {}", winamp_config.sa, mode);
    }

    let sa_peaks_str = format!("sa_peaks={}", winamp_config.sa_peaks.unwrap_or_default());
    if config_content.contains(&sa_peaks_str) {
        config_content = config_content.replace(&sa_peaks_str, &format!("sa_peaks={}", *peaks.lock().unwrap()));
        //println!("sa_peaks: {:?}, peaks: {}", winamp_config.sa_peaks, new_value);
    }

    let vu_peaks_str = format!("vu_peaks={}", winamp_config.vu_peaks.unwrap_or_default());
    if config_content.contains(&vu_peaks_str) {
        config_content = config_content.replace(&vu_peaks_str, &format!("vu_peaks={}", *vu_peaks.lock().unwrap()));
        //println!("sa_peaks: {:?}, peaks: {}", winamp_config.sa_peaks, new_value);
    }

    let vu_peak_fall_off_str = format!("vu_peak_fall_off={}", winamp_config.vu_peak_fall_off.unwrap_or_default());
    if config_content.contains(&vu_peak_fall_off_str) {
        config_content = config_content.replace(&vu_peak_fall_off_str, &format!("vu_peak_fall_off={}", *vupeakfo - 1));
        //println!("sa_peaks: {:?}, peaks: {}", winamp_config.sa_peaks, new_value);
    }

    let prefs_tab_str = format!("prefs_tab={}", winamp_config.prefs_tab.unwrap_or_default());
    if config_content.contains(&prefs_tab_str) {
        config_content = config_content.replace(&prefs_tab_str, &format!("prefs_tab={}", *prefs_id));
        //println!("sa_peaks: {:?}, peaks: {}", winamp_config.sa_peaks, new_value);
    }

    let aot_str = format!("aot={}", winamp_config.aot.unwrap_or_default());
    if config_content.contains(&aot_str) {
        config_content = config_content.replace(&aot_str, &format!("aot={}", *aot.lock().unwrap()));
        //println!("sa_peaks: {:?}, peaks: {}", winamp_config.sa_peaks, new_value);
    }

    let vu_rms_str = format!("vu_rms={}", winamp_config.vu_rms.unwrap_or_default());
    if config_content.contains(&vu_rms_str) {
        config_content = config_content.replace(&vu_rms_str, &format!("vu_rms={}", *vu_rms - 1));
        //println!("sa_peaks: {:?}, peaks: {}", winamp_config.sa_peaks, new_value);
    }
      
    // Update config_safire based on user input and other conditions
    if *bandwidth.lock().unwrap() == "thin" {
        // Set bandwidth to thin bands
        safire |= 32; // Set the 6th bit to 1
    } else {
        // Set bandwidth to thick bands (default)
        safire &= !32; // Set the 6th bit to 0
    }

    match oscstyle.lock().unwrap().as_str() {
        "dots" => safire = safire & !(3 << 2),
        "lines" => safire = (safire & !(3 << 2)) | (1 << 2),
        "solid" => safire = (safire & !(3 << 2)) | (2 << 2),
        _ => println!("Invalid oscilloscope style."),
    }

    match specdraw.lock().unwrap().as_str() {
        "normal" => safire = safire & !3,
        "fire" => safire = (safire & !3) | 1,
        "line" => safire = (safire & !3) | 2,
        _ => println!("Invalid analyzer style."),
    }

    match vu_style.lock().unwrap().as_str() {
        "normal" => *vu_style_mutex = 0,
        "fire" => *vu_style_mutex = 1,
        "line" => *vu_style_mutex = 2,
        _ => println!("Invalid analyzer style."),
    }

    // Update config_content based on the modified config_safire
    let safire_str = format!("safire={}", winamp_config.safire.unwrap_or_default());
    if config_content.contains(&safire_str) {
        config_content = config_content.replace(&safire_str, &format!("safire={}", safire));
        //println!("wac_safire: {:?}, config_safire: {}", winamp_config.safire, safire);
    }

    let vu_style_str = format!("vu_style={}", winamp_config.vu_style.unwrap_or_default());
    if config_content.contains(&vu_style_str) {
        config_content = config_content.replace(&vu_style_str, &format!("vu_style={}", vu_style_mutex));
        //println!("wac_vu_style: {:?}, vu_style: {}", winamp_config.vu_style, vu_style_mutex);
    }

    // Check if any changes were made before writing to the file
    if config_content != fs::read_to_string(config_file_path.clone()).unwrap_or_default() {
        match fs::write(config_file_path, config_content) {
            Ok(_) => {
                // Configuration updated successfully, no need to print anything
            }
            Err(_) => {
                // Error updating configuration, no need to print anything
            }
        }
    } else {
        // No changes to the configuration, no need to print anything
    }

    Ok(())
}

fn draw_diag(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    ys: &[f32],
    fft: &[f32],
) -> Result<(), String> {
    let rect = Rect::new(0, 0, 606, 592);
    canvas.set_draw_color(cgenex[2]);
    canvas.fill_rect(rect).unwrap();

    let ys: Vec<i32> = ys.iter().step_by(8).map(|&sample| ((sample as f32 + 128.0)) as i32).collect();
    let xs: Vec<i32> = (0..256).collect();
    let fft: Vec<i32> = fft.iter()
    .map(|&sample| ((sample as f32)) as i32)
    .collect(); // cast to i32
    canvas.set_draw_color(cgenex[5]);
    let rect = Rect::new(185, 28, 412, 556);
    canvas.draw_rect(rect).unwrap();

    let mut last_y = 0;
    let mut top: i32;
    let mut bottom: i32;

    for (x, y) in xs.iter().zip(ys.iter()) {
        let x = *x;
        let y = *y;
        if x == 0 {
            last_y = y;
        }
        let half_osc = 128;
        if y >= half_osc{
            top = half_osc;
            bottom = y;
        } else {
            top = y;
            bottom = half_osc-1;
        }
            for dy in top..=bottom {
                let point = Point::new(x + 230, dy/2);
                canvas.set_draw_color(Color::RGB(255, 255, 255));
                canvas.draw_point(point).unwrap();
            }
        }

        for (x, y) in xs.iter().zip(fft.iter().step_by(4)) {
            let x = *x;
            let y = *y;
            if x == 0 {
                last_y = y;
            }
            top = last_y;
            bottom = y;
            last_y = y;
            if bottom < top {
                std::mem::swap(&mut bottom, &mut top);
                top += 1;
            }
                for dy in top..=bottom {
                    let point = Point::new(x + 230, -dy+400);
                    canvas.set_draw_color(Color::RGB(255, 255, 255));
                    canvas.draw_point(point).unwrap();
                }
            }

    Ok(())
}

fn get_device_name_by_index(index: usize) -> Option<String> {
    let host = cpal::default_host();
    let input_devices = host.input_devices().ok()?;
    let output_devices = host.output_devices().ok()?;
    let all_devices = input_devices.chain(output_devices);

    for (device_index, device) in all_devices.enumerate() {
        if device_index == index {
            return Some(device.name().unwrap_or_else(|_| "Unknown Device".to_string()));
        }
    }

    None
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let mut device_in_use: String = "".to_string();

    let selected_device_index = match args.device {
        Some(index) => {
            // COMMAND LINE ARGUMENT
            if index == 0 {
                eprintln!("Device index should start from 1.");
                std::process::exit(1);
            }
    
            if let Some(device_name) = get_device_name_by_index(index - 1) {
                device_in_use = format!("{}. {}", index, device_name);
            } else {
                eprintln!("Device with index {} not found.", index);
                std::process::exit(1);
            }
    
            index - 1
        }
        None => {
            // Prompt the user to select an audio device
            let selection = prompt_for_device();
            if selection.is_none() {
                std::process::exit(1);
            }

            // Access the selected index and name
            let selected_device_index = selection.as_ref().unwrap().index;
            let selected_device_name = &selection.as_ref().unwrap().name;
            device_in_use = format!("{}. {}", selected_device_index + 1, selected_device_name);

            selected_device_index
        }
    };

    //println!("{}", device_in_use);

    let mut debug_vector: Vec<f64> = vec![0.0; 75];
    debug_vector[0] = 15.0;
    
    // handle args
    let oscstyle = Arc::new(Mutex::new("lines".to_string())); // Convert String to &str
    let specdraw = Arc::new(Mutex::new("normal".to_string()));
    let vudraw = Arc::new(Mutex::new("normal".to_string()));
    let zoom = args.zoom;
    let amp = Arc::new(Mutex::new(1));
    let mut mode: u8;
    let bandwidth = Arc::new(Mutex::new("thick".to_string()));
    let peakfo = Arc::new(Mutex::new(1));
    let barfo = Arc::new(Mutex::new(2));
    let peaks = Arc::new(Mutex::new(1));
    let vu_peaks = Arc::new(Mutex::new(1));
    let vu_style_num = Arc::new(Mutex::new(0));
    let vu_peak_fall_off = Arc::new(Mutex::new(2));
    let rms_v = Arc::new(Mutex::new(4));
    let prefs_tab_id = Arc::new(Mutex::new(0));
    let aot: Arc<Mutex<u8>> = Arc::new(Mutex::new(0));

    let mut peakfo_unlocked = peakfo.lock().unwrap();
    let mut barfo_unlocked = barfo.lock().unwrap();
    let mut amp_unlocked = amp.lock().unwrap();
    let mut rms_v_unlocked = rms_v.lock().unwrap();

    let mut vupeakfo_unlocked = vu_peak_fall_off.lock().unwrap();

    let mut prefs_tab_unlocked = prefs_tab_id.lock().unwrap();

    if args.debug == true {
        println!("debug");
    }

    if *peakfo_unlocked <= 1 {
        *peakfo_unlocked = 1;
    } else if *peakfo_unlocked >= 5 {
        *peakfo_unlocked = 5;
    }

    if *barfo_unlocked <= 1 {
        *barfo_unlocked = 1;
    } else if *barfo_unlocked >= 5 {
        *barfo_unlocked = 5;
    }

    if *vupeakfo_unlocked <= 1 {
        *vupeakfo_unlocked = 1;
    } else if *vupeakfo_unlocked >= 5 {
        *vupeakfo_unlocked = 5;
    }

    if *rms_v_unlocked <= 1 {
        *rms_v_unlocked = 1;
    } else if *rms_v_unlocked >= 5 {
        *rms_v_unlocked = 5;
    }

    // Extract the configuration file path from the command-line arguments
    let config_file_path = args.configini;

    // Read the content of the configuration file
    let config_content = match fs::read_to_string(config_file_path.clone()) {
        Ok(content) => content,
        Err(_) => {
            eprintln!("Error reading the configuration file.");
            return Err("Failed to read configuration file.".to_string());
        }
    };
    
    // Parse the configuration file content
    let mut winamp_config = winamp_ini(&config_content);
    
    // Access the values in the configuration
    let sa = winamp_config.sa.unwrap_or_default();
    let safire = winamp_config.safire.unwrap_or_default();
    let sa_peaks = winamp_config.sa_peaks.unwrap_or_default();
    let safalloff = winamp_config.safalloff.unwrap_or_default();
    let sa_peak_falloff = winamp_config.sa_peak_falloff.unwrap_or_default();
    let sa_amp = winamp_config.sa_amp.unwrap_or_default();
    let vupeaks = winamp_config.vu_peaks.unwrap_or_default();
    let vu_peak_fall_off_c = winamp_config.vu_peak_fall_off.unwrap_or_default();
    let vu_style = winamp_config.vu_style.unwrap_or_default();
    let vu_rms = winamp_config.vu_rms.unwrap_or_default();
    let prefs_tab = winamp_config.prefs_tab.unwrap_or_default();
    let aotc = winamp_config.aot.unwrap_or_default();
    *vu_style_num.lock().unwrap() = vu_style;
    *prefs_tab_unlocked = prefs_tab;
    *aot.lock().unwrap() = aotc as u8;

    let mut aot_bool = false;

    *barfo_unlocked = safalloff as u8 + 1;
    *peakfo_unlocked = sa_peak_falloff as u8 + 1;
    mode = sa as u8;
    *amp_unlocked = sa_amp as u8;
    *peaks.try_lock().unwrap() = sa_peaks as u8;
    *vupeakfo_unlocked = vu_peak_fall_off_c as u8 + 1;
    *vu_peaks.try_lock().unwrap() = vupeaks as u8;
    *rms_v_unlocked = vu_rms as u8 + 1;

    if (safire & (1 << 5)) != 0 {
        *bandwidth.lock().unwrap() = "thin".to_string();
    }
    match (safire >> 2) & 3 {
        0 => *oscstyle.lock().unwrap() = "dots".to_string(),
        1 => *oscstyle.lock().unwrap() = "lines".to_string(),
        2 => *oscstyle.lock().unwrap() = "solid".to_string(),
        _ => println!("Invalid oscilloscope style in config_safire."),
    }
    println!("Loaded oscilloscope style: {}", oscstyle.lock().unwrap().as_str());

    match safire & 3 {
        0 => *specdraw.lock().unwrap() = "normal".to_string(),
        1 => *specdraw.lock().unwrap() = "fire".to_string(),
        2 => *specdraw.lock().unwrap() = "line".to_string(),
        _ => println!("Invalid analyzer style in config_safire."),
    }
    println!("Loaded Spectrum Analyzer style: {}", specdraw.lock().unwrap().as_str());

    match vu_style & 3 {
        0 => *vudraw.lock().unwrap() = "normal".to_string(),
        1 => *vudraw.lock().unwrap() = "fire".to_string(),
        2 => *vudraw.lock().unwrap() = "line".to_string(),
        _ => println!("Invalid analyzer style in config_safire."),
    }
    println!("Loaded VU Meter style: {}", *vudraw.lock().unwrap());
    //config_sa_peaks = *peaks_unlocked != 0;

    let mut bars = [Bar {
        height: 0,
        height2: 0.0,
        peak: 0.0,
        gravity: 0.0,
        bargrav: 0.0,
        vumeter: 0,
        vumeterpeak: 0.0,
    }; NUM_BARS];

    let mut mouse_x: i32 = 0;
    let mut mouse_y: i32 = 0;

    // set up sdl2
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;
    //sdl2::hint::set("SDL_HINT_RENDER_SCALE_QUALITY", "0");

    let window2 = video_subsystem
        .window("RustampVis Preferences", 606 as u32, 592 as u32)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas2 = window2.into_canvas().build().unwrap();

    let window = video_subsystem
        .window("Winamp Mini Visualizer (in Rust)", (WINDOW_WIDTH * zoom) as u32, (WINDOW_HEIGHT * zoom) as u32)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();

    let ttf_context = sdl2::ttf::init().unwrap();
    let texture_creator = canvas2.texture_creator();
    let font_path: &str;
    let font_path_bold: &str;
    let vectorgfx_path: &str;
    let vectorgfx_size: u16;
    
    if cfg!(windows) {
        font_path = "C:\\Windows\\fonts\\tahoma.ttf";
        font_path_bold = "C:\\Windows\\fonts\\tahomabd.ttf";
        vectorgfx_path = "C:\\Windows\\fonts\\marlett.ttf";
        vectorgfx_size = 15;
    } else if cfg!(unix) {
        font_path = "font/tahoma.ttf";
        font_path_bold = "font/tahomabd.ttf";
        vectorgfx_path = "font/marlett.ttf";
        vectorgfx_size = 15;
    } else {
        // Handle the case where neither windows nor unix is the configuration
        return Err("Unsupported platform".to_string());
    }
    
    let mut font = ttf_context.load_font(font_path, 11)
        .map_err(|err| format!("failed to load font: {}", err))?;
    let mut font_bold = ttf_context.load_font(font_path_bold, 11)
    .map_err(|err| format!("failed to load font: {}", err))?;
    let mut vectorgfx = ttf_context.load_font(vectorgfx_path, vectorgfx_size)
        .map_err(|err| format!("failed to load vector graphics font: {}", err))?;
    
    font.set_hinting(sdl2::ttf::Hinting::Mono);
    font_bold.set_hinting(sdl2::ttf::Hinting::Mono);
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
    let vuc = vucolor(&viscolors);

    let audio_data = Arc::new(Mutex::new(vec![0.0; 4096]));
    let audio_data_l = Arc::new(Mutex::new(vec![0.0; 4096]));
    let audio_data_r = Arc::new(Mutex::new(vec![0.0; 4096]));
    let spec_data = Arc::new(Mutex::new(vec![0.0; 4096]));

    // Create a blocking receiver to get audio samples from the audio stream loop.
    let (tx, rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
    let (tx_l, rx_l): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
    let (tx_r, rx_r): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
    let (s, r): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();

    let audio_data_clone = Arc::clone(&audio_data);
    let audio_data_clone_l = Arc::clone(&audio_data_l);
    let audio_data_clone_r = Arc::clone(&audio_data_r);
    let spec_data_clone = Arc::clone(&spec_data);

    // Start the audio stream loop in a separate thread.
    thread::spawn(move || audio_stream_loop(tx, tx_l, tx_r, s, Some(selected_device_index)));

    thread::spawn(move || {
        loop {
            // swap the captured audio samples with the visualization data.
            if let Ok(audio_samples) = rx.recv() {
                let mut audio_data = audio_data_clone.lock().unwrap();
                *audio_data = audio_samples;
            }

            if let Ok(audio_samples_l) = rx_l.recv() {
                let mut audio_data_l = audio_data_clone_l.lock().unwrap();
                *audio_data_l = audio_samples_l;
            }

            if let Ok(audio_samples_r) = rx_r.recv() {
                let mut audio_data_r = audio_data_clone_r.lock().unwrap();
                *audio_data_r = audio_samples_r;
            }

            if let Ok(spec_samples) = r.recv() {
                let mut spec_data = spec_data_clone.lock().unwrap();
                *spec_data = spec_samples;
            }

            std::thread::sleep(std::time::Duration::from_millis(0));
        }
    });

    let image_path = "gen_ex.png";
    let genex_colors = genex(image_path);
    let mut is_button_clicked = false;

    let mut scroll: f64 = 0.0;

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
                Event::KeyDown { window_id: 2, keycode: Some(Keycode::R), .. } => {
                    // Reload the viscolors data when "R" key is pressed
                    let new_viscolors = viscolors::load_colors(&args.viscolor);
                    let new_osc_colors = osccolors(&new_viscolors);
                    let new_peakrgb = peakc(&new_viscolors);
                    viscolors = new_viscolors;
                    osc_colors = new_osc_colors;
                    peakrgb = new_peakrgb;
                }
                Event::KeyDown { window_id: 2, keycode: Some(Keycode::B), .. } => {
                    // switch bandwidth
                    let mut bandwidth = bandwidth.lock().unwrap();
                    if *bandwidth == "thick" {
                        switch_bandwidth(&mut *bandwidth);
                    } else if *bandwidth == "thin" {
                        switch_bandwidth(&mut *bandwidth);
                    }
                }
                Event::KeyDown { window_id: 2, keycode: Some(Keycode::A), .. } => {
                    aot_bool = !aot_bool;
                    canvas.window_mut().set_always_on_top(aot_bool);
                    if aot_bool {
                        *aot.lock().unwrap() = 1;
                    }
                    else {
                        *aot.lock().unwrap() = 0;
                    }
                }

                Event::MouseButtonDown { window_id: 2, mouse_btn: MouseButton::Right, .. } => {
                    let mut oscstyle = oscstyle.lock().unwrap();
                    let mut specdraw = specdraw.lock().unwrap();
                    let mut vudraw = vudraw.lock().unwrap();
                    if mode == 2 {
                        switch_oscstyle(&mut *oscstyle);
                    } else if mode == 1 {
                        switch_specstyle(&mut *specdraw);
                    } else if mode == 3 {
                        switch_vustyle(&mut *vudraw);
                    }
                }
                Event::MouseButtonDown { window_id: 2, mouse_btn: MouseButton::Left, .. } => {
                    mode = (mode + 1) % 4;
                    //println!("{mode}")
                }
                Event::MouseMotion { window_id: 1, x, y, .. } => {
                    mouse_x = x;
                    mouse_y = y;
                    // Handle mouse motion events
                    //println!("Mouse moved to ({}, {})", x, y);
                }
                Event::MouseButtonDown { window_id: 1, mouse_btn, x: _, y: _, .. } => {
                    // Handle mouse button down events
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        is_button_clicked = true;
                        //println!("{:?} {:?}", barfo_unlocked, peakfo_unlocked);
                    }
                }
                Event::MouseButtonUp { window_id: 1, mouse_btn, x: _, y: _, .. } => {
                    // Handle mouse button down events
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        is_button_clicked = false;
                    }
                }
                _ => {}
            }
        }

        let audio_data = audio_data.lock().unwrap().clone();
        let audio_data_l = audio_data_l.lock().unwrap().clone();
        let audio_data_r = audio_data_r.lock().unwrap().clone();
        let spec_data = spec_data.lock().unwrap().clone();

        //println!("Captured audio samples: {:?}", audio_data);

        canvas.window_mut().set_always_on_top(*aot.lock().unwrap() != 0);

        //println!("{}", sdl2::get_framerate());
        draw_visualizer(&mut canvas, &viscolors, &osc_colors, &vuc, peakrgb, &audio_data, &audio_data_l, &audio_data_r, &spec_data, &*oscstyle.lock().unwrap(), &*specdraw.lock().unwrap(), &*vudraw.lock().unwrap(), mode, &*bandwidth.lock().unwrap(), zoom, &mut bars, *peakfo_unlocked, *vupeakfo_unlocked, *barfo_unlocked, *rms_v_unlocked, peaks.clone(), vu_peaks.clone(), *amp_unlocked, args.debug, debug_vector.clone());
        draw_window(
            &mut canvas2,
            &genex_colors,
            &texture_creator,
            &font,
            &font_bold,
            &vectorgfx,
            oscstyle.clone(),
            mode,
            image_path,
            is_button_clicked,
            &mut mouse_x,
            &mut mouse_y,
            peaks.clone(),
            vu_peaks.clone(),
            specdraw.clone(),
            vudraw.clone(),
            vu_style_num.clone(),
            bandwidth.clone(),
            &mut *peakfo_unlocked,
            &mut *vupeakfo_unlocked,
            &mut *barfo_unlocked,
            &mut rms_v_unlocked,
            &mut *amp_unlocked,
            scroll,
            &mut bars,
            config_content.clone(),
            &mut winamp_config,
            config_file_path.clone(),
            safire,
            &device_in_use,
            &mut prefs_tab_unlocked,
            aot.clone(),
        )?;
        
        //draw_diag(&mut canvas2, &genex_colors, &*audio_data, &*spec_data)?;

        // draw the cool shit
        canvas.present();
        canvas2.present();
        scroll += SCROLL_SPEED;

        std::thread::sleep(std::time::Duration::from_millis(13));
    }

    // Stop the audio streaming loop gracefully
    //audio_thread.join().unwrap();
    //gracefully my ass ChatGPT, this shit hung the entire thing on closing
    Ok(())
}

fn prompt_for_device() -> Option<DeviceSelection> {
    let host = cpal::default_host();
    let devices = host.devices().expect("Failed to retrieve devices").collect::<Vec<_>>();
    
    println!("Available audio devices:");
    for (index, device) in devices.iter().enumerate() {
        println!("{}. {}", index + 1, device.name().unwrap_or("Unknown Device".to_string()));
    }
    
    println!("Enter the number of the audio device (Speakers or Microphone) to visualize: ");

    loop {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).expect("Failed to read line");

        if let Ok(index) = input.trim().parse::<usize>() {
            if index > 0 && index <= devices.len() {
                let selected_device = &devices[index - 1];
                let device_name = selected_device.name().unwrap_or("Unknown Device".to_string());
                // Print debug information
                //println!("{}. {}", index, device_name);
                
                // Return a struct with both the index and name
                return Some(DeviceSelection { index: index - 1, name: device_name });
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

fn vucolor(colors: &[Color]) -> Vec<Color> {
    vec![
        colors[17],
        colors[17],
        colors[17],
        colors[17],
        colors[17],
        colors[16],
        colors[16],
        colors[16],
        colors[16],
        colors[16],
        colors[15],
        colors[15],
        colors[15],
        colors[15],
        colors[14],
        colors[14],
        colors[14],
        colors[14],
        colors[14],
        colors[13],
        colors[13],
        colors[13],
        colors[13],
        colors[13],
        colors[12],
        colors[12],
        colors[12],
        colors[12],
        colors[12],
        colors[11],
        colors[11],
        colors[11],
        colors[11],
        colors[10],
        colors[10],
        colors[10],
        colors[10],
        colors[10],
        colors[9],
        colors[9],
        colors[9],
        colors[9],
        colors[9],
        colors[8],
        colors[8],
        colors[8],
        colors[8],
        colors[8],
        colors[7],
        colors[7],
        colors[7],
        colors[7],
        colors[6],
        colors[6],
        colors[6],
        colors[6],
        colors[6],
        colors[5],
        colors[5],
        colors[5],
        colors[5],
        colors[5],
        colors[4],
        colors[4],
        colors[4],
        colors[4],
        colors[4],
        colors[3],
        colors[3],
        colors[3],
        colors[3],
        colors[2],
        colors[2],
        colors[2],
        colors[2],
    ]
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
