//ChatGPT assisted me with this.
//Base code came from https://stackoverflow.com/questions/72181090/getting-pixels-of-many-different-colors-when-drawing-with-sdl-in-rust

extern crate sdl2;

use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use std::sync::{Arc, Mutex};
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
const ZOOM: i32 = 7;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Oscilloscope style
    #[arg(short, long, default_value = "lines")]
    oscstyle: String, // Change this to String

    /// Name of the custom viscolor.txt file
    #[arg(short, long, default_value = "viscolor.txt")]
    viscolor: String,
}

fn draw_oscilloscope(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    _colors: &[Color],
    osc_colors: &[Color],
    ys: &[u8],
    oscstyle: &str,
) {
    let xs: Vec<i32> = (0..WINDOW_WIDTH).collect();
    let ys: Vec<i32> = ys.iter().map(|&sample| (sample as i32 / 8) - 9).collect(); // cast to i32

    let mut last_y = 0;

    for x in 0..WINDOW_WIDTH {
        for y in 0..WINDOW_HEIGHT {
            if x % 2 == 1 || y % 2 == 0 {
                let rect = Rect::new(x * ZOOM, y * ZOOM, ZOOM as u32, ZOOM as u32);
                canvas.set_draw_color(_colors[0]);
                canvas.fill_rect(rect).unwrap();
            } else {
                let rect = Rect::new(x * ZOOM, y * ZOOM, ZOOM as u32, ZOOM as u32);
                canvas.set_draw_color(_colors[1]);
                canvas.fill_rect(rect).unwrap();
            }
        }
    }
    for (x, y) in xs.iter().zip(ys.iter()) {
        let x = *x;
        let y = *y;

        let x = std::cmp::min(std::cmp::max(x, 0), WINDOW_WIDTH - 1);
        let y = std::cmp::min(std::cmp::max(y, 0), (WINDOW_HEIGHT - 1).try_into().unwrap());

        if x == 0 {
            last_y = y;
        }

        let mut top = y;
        let mut bottom = last_y;
        last_y = y;

        if oscstyle == "lines" {
            if bottom < top {
                std::mem::swap(&mut bottom, &mut top);
                top += 1;
            }
    
            for dy in top..=bottom {
                let color_index = (top as usize) % osc_colors.len();
                let scope_color = osc_colors[color_index];
                let rect = Rect::new(x * ZOOM, dy * ZOOM, ZOOM as u32, ZOOM as u32);
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
                let rect = Rect::new(x * ZOOM, dy * ZOOM, ZOOM as u32, ZOOM as u32);
                canvas.set_draw_color(scope_color);
                canvas.fill_rect(rect).unwrap();
            }
        } else if oscstyle == "dots" {
            for _dy in -1..y {
                let color_index = (y as usize) % osc_colors.len();
                let scope_color = osc_colors[color_index];
                let rect = Rect::new(x * ZOOM, y * ZOOM, ZOOM as u32, ZOOM as u32);
                canvas.set_draw_color(scope_color);
                canvas.fill_rect(rect).unwrap();
            }
        } else {
            eprintln!("Invalid oscilloscope style. Supported styles: lines, solid, dots.");
            // You can handle this error case according to your needs
            // ...
        }
    }
}

fn audio_stream_loop(tx: Sender<Vec<u8>>) {
    enum ConfigType {
        Windows(cpal::SupportedStreamConfig),
        Unix(cpal::StreamConfig),
    }
    let host = cpal::default_host();
    let device = host.default_input_device().expect("failed to find input device");
    let config: ConfigType; // Declare the config variable here

    if cfg!(windows) {
        config = ConfigType::Windows(device.default_input_config().expect("Failed to get default input config"));
    } else if cfg!(unix) {
        config = ConfigType::Unix(cpal::StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(44100),
            buffer_size: cpal::BufferSize::Fixed(2048), //not 1152 or 576 but eh, visually it looks *close* enough.
        });
    } else {
        panic!("Unsupported platform");
    }

    // Create a ring buffer (VecDeque) with a maximum capacity of 576 samples
    let mut ring_buffer: VecDeque<u8> = VecDeque::with_capacity(75); //HAHA SCREW YOU WASAPI, NOW YOU WILL NOT COMPLAIN

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Convert f32 samples to u8 (0-255) and collect them into a Vec<u8>
        let left_channel_samples: Vec<u8> = data
            .iter()
            .step_by(14) // Skip every other sample (right channel)
            .map(|&sample| ((-sample + 1.03) * 127.5) as u8)
            .collect();

        // Extend the ring buffer with the new samples
        for sample in &left_channel_samples {
            if ring_buffer.len() == ring_buffer.capacity() {
                ring_buffer.pop_front(); // Remove the oldest sample when the buffer is full
            }
            ring_buffer.push_back(*sample);
        }

        // Convert the ring buffer to a regular Vec<u8> and send it through the channel
        tx.send(ring_buffer.iter().copied().collect()).unwrap();
    };

    // When creating the stream, pattern match on the ConfigType to get the appropriate config
    let stream = match config {
        ConfigType::Windows(conf) => device.build_input_stream(&conf.into(), callback, err_fn, None),
        ConfigType::Unix(conf) => device.build_input_stream(&conf.into(), callback, err_fn, None),
    }
    .unwrap();
    stream.play().unwrap();

    // The audio stream loop should not block, so we use an empty loop.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(32768));
    }
}


fn main() -> Result<(), anyhow::Error> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    //let args: Vec<String> = env::args().collect();

    let args = Args::parse();

    // Extract the oscstyle field from Args struct
    let oscstyle = args.oscstyle.as_str(); // Convert String to &str
    let window = video_subsystem
        .window("Winamp Mini Visualizer (in Rust)", (WINDOW_WIDTH * ZOOM) as u32, (WINDOW_HEIGHT * ZOOM) as u32)
        .position_centered()
        .build()
        .unwrap();

    // Load the custom viscolor.txt file
    let viscolors = viscolors::load_colors(&args.viscolor);
    let osc_colors = osc_colors_and_peak(&viscolors);

    let mut canvas = window.into_canvas().build().unwrap();

    let mut event_pump = sdl_context.event_pump().unwrap();

    // The vector to store captured audio samples.
    let audio_data = Arc::new(Mutex::new(Vec::<u8>::new()));

    // Create a blocking receiver to get audio samples from the audio stream loop.
    let (tx, rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = unbounded();

    // Start the audio stream loop in a separate thread.
    thread::spawn(move || audio_stream_loop(tx));

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        // Lock the mutex and swap the captured audio samples with the visualization data.
        let audio_samples = rx.recv().unwrap();
        //println!("Captured audio samples: {:?}", audio_samples);

        // Lock the mutex and update the captured audio samples.
        let mut audio_data = audio_data.lock().unwrap();
        *audio_data = audio_samples;
        //println!("Captured audio samples: {:?}", audio_data);

        draw_oscilloscope(&mut canvas, &viscolors, &osc_colors, &*audio_data, oscstyle/* , modern*/);

        canvas.present();

        std::thread::sleep(std::time::Duration::from_millis(0));
    }

    // Stop the audio streaming loop gracefully
    //audio_thread.join().unwrap();
    //gracefully my ass ChatGPT, this shit hung the entire thing on closing
    Ok(())
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