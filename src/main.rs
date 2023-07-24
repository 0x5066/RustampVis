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

mod viscolors;

const WINDOW_WIDTH: i32 = 75;
const WINDOW_HEIGHT: i32 = 16;
const ZOOM: i32 = 7;

fn draw_oscilloscope(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    _colors: &[Color],
    osc_colors: &[Color],
    ys: &[u8],
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
    }
}

fn audio_stream_loop(tx: Sender<Vec<u8>>) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("failed to find input device");
    let streamc = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(44100),
        buffer_size: cpal::BufferSize::Fixed(2048),
    };

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

        // Send audio samples through the channel
        tx.send(left_channel_samples).unwrap();
    };

    let stream = device.build_input_stream(&streamc.into(), callback, err_fn, None).unwrap();
    stream.play().unwrap();

    // The audio stream loop should not block, so we use an empty loop.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(32768));
    }
}


fn main() -> Result<(), anyhow::Error> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("Winamp Mini Visualizer (in Rust)", (WINDOW_WIDTH * ZOOM) as u32, (WINDOW_HEIGHT * ZOOM) as u32)
        .position_centered()
        .build()
        .unwrap();

    let viscolors = viscolors::load_colors("viscolor.txt");
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

        draw_oscilloscope(&mut canvas, &viscolors, &osc_colors, &*audio_data);

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