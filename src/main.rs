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
const NUM_BARS: usize = 75;
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
    bars: &mut [Bar],
) {
    let xs: Vec<i32> = (0..WINDOW_WIDTH).collect();
    let ys: Vec<i32> = ys.iter().step_by(16).map(|&sample| ((sample as i32 / 8) - 9)/* * WINDOW_HEIGHT / 16*/).collect(); // cast to i32
    let fft: Vec<i32> = fft.iter().step_by(11).map(|&sample| ((sample as i32 / 8) - 9)).collect(); // cast to i32

    //let mut bars: Vec<Bar> = vec![Bar::new(); NUM_BARS];
    let mut last_y = 0;
    let mut top: i32 = 0; //bro it is being read though wth?!
    let mut bottom: i32 = 0;
    let fft_value: f64 = 0.0;

    // Process your FFT data and store it in the respective Bar instances
    for (bar, &fft_value) in bars.iter_mut().zip(fft.iter()) {
        // Access the values from bars and fft here, which are represented by (bar, fft_value).
        // You can use 'bar' and 'fft_value' as needed in this section of the loop.
    
        // For example, if you want to update 'bar.height' based on 'fft_value':
        bar.height = fft_value as f64 + 9.0;
        if bar.height >= 15.0{
            bar.height = 15.0;
        }
    
    }
        // Draw the bars

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
if mode == 0 {

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
                // You can handle this error case according to your needs
                // ...
            }
        }

    } else if mode == 0{
        for (i, bar) in bars.iter().enumerate() {
            let x = i as i32 * zoom as i32;
            let y = -bar.height2 as i32 + 15;

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
                let rect = Rect::new(x, dy * zoom, zoom as u32, zoom as u32);
                canvas.set_draw_color(color);
                canvas.fill_rect(rect).unwrap();
            }
        }
        for (i, bar) in bars.iter().enumerate() {
            let bar_x = i as i32 * zoom as i32;
            let bar_height = -bar.peak + 15.0;
            let mut peaki32: i32 = bar_height as i32;
	    //let bar_height2 = -bar.height as i32 + 15;
        //println!("{}", bar_height);

        if peaki32 > 14 {
            peaki32 = 18;
        }

            let rect = Rect::new(
                bar_x, peaki32 * zoom as i32,
                zoom as u32,
                zoom as u32,
            );
            canvas.set_draw_color(_colors[23]);
            canvas.fill_rect(rect).unwrap();
	}
        for i in 0..NUM_BARS {
            if bars[i].height2 > bars[i].peak {
                bars[i].gravity = 0.0;
                bars[i].peak = bars[i].height2;
                
            } else {
                if bars[i].gravity <= 2.0 {
                    bars[i].gravity += 0.007;
                }
                bars[i].peak = if bars[i].peak <= 0.0 {
                    0.0
                } else {
                    bars[i].peak - bars[i].gravity
                };
            } 
            bars[i].height2 -= bars[i].bargrav*6.0;
            // Print the values
            /*println!(
                "Bar {} - Height: {}, Peak: {}, Gravity: {}",
                i + 1,
                bars[i].height,
                bars[i].peak,
                bars[i].gravity
            );*/

            if bars[i].height2 <= bars[i].height {
                bars[i].height2 = bars[i].height;
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
        spectrum[0].im = 0.0;

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

    let mut bars = [Bar {
        height: 0.0,
        height2: 0.0,
        peak: 0.0,
        gravity: 0.2,
        bargrav: 0.2,
    }; NUM_BARS];

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

        draw_oscilloscope(&mut canvas, &viscolors, &osc_colors, &*audio_data, &*spec_data, oscstyle, specdraw, mode, zoom, &mut bars/* , modern*/);

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