//ChatGPT assisted me with this.
//Base code came from https://stackoverflow.com/questions/72181090/getting-pixels-of-many-different-colors-when-drawing-with-sdl-in-rust

extern crate sdl2;

use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use anyhow;

mod viscolors;

const WINDOW_WIDTH: i32 = 75;
const WINDOW_HEIGHT: i32 = 16;
const ZOOM: i32 = 5;

fn draw_oscilloscope(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    _colors: &[Color],
    osc_colors: &[Color],
    ys: &[i32],
) {

    // Find the maximum absolute value of the audio samples for scaling
    let max_sample = ys
        .iter()
        .map(|&sample| sample.abs())
        .max()
        .unwrap_or(1); // Avoid division by zero

    let _ys_data: Vec<i32> = ys
        .iter()
        .map(|&sample| {
            let scaled_sample = (sample as f32 * WINDOW_HEIGHT as f32 / 2.0 / max_sample as f32)
                as i32;
            scaled_sample + (WINDOW_HEIGHT-7)
        })
        .collect();
    println!("{_ys_data:?}");
    let xs: Vec<i32> = (0..WINDOW_WIDTH * 4).collect();
    let ys: Vec<i32> = _ys_data;

    let mut last_y = 0;

    for (x, y) in xs.iter().zip(ys.iter()) {
        let x = *x;
        let y = *y;

        let x = std::cmp::min(std::cmp::max(x, 0), WINDOW_WIDTH - 1);
        let y = std::cmp::min(std::cmp::max(y, 0), WINDOW_HEIGHT - 1);

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

fn main() -> Result<(), anyhow::Error> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("rust-sdl2 demo", 600, 128)
        .position_centered()
        .build()
        .unwrap();

    let viscolors = viscolors::load_colors("viscolor.txt");
    let osc_colors = osc_colors_and_peak(&viscolors);

    let mut canvas = window.into_canvas().build().unwrap();
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let mut event_pump = sdl_context.event_pump().unwrap();

    // The vector to store captured audio samples.
    let ys = Arc::new(Mutex::new(Vec::<i32>::new()));

    // Create a reference to ys for the callback function to store captured samples.
    let ys_callback = ys.clone();

    // Run the input stream on a separate thread.
    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("failed to find input device");
        let config = device
            .default_input_config()
            .expect("Failed to get default input config");

        let err_fn = move |err| {
            eprintln!("an error occurred on stream: {}", err);
        };

        let callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut ys = ys_callback.lock().unwrap();
            let ys_i32: Vec<i32> = data.iter().map(|&sample| (sample * i32::MAX as f32) as i32).collect();
            ys.extend_from_slice(&ys_i32);            
            //println!("{data:?}");
        };

        let stream = device.build_input_stream(&config.into(), callback, err_fn, None).unwrap();
        stream.play().unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1));
        //stream.pause().unwrap();
        //stream.into_inner();
        //println!("Recording complete!");
    });

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();

        let ys = ys.lock().unwrap();
        draw_oscilloscope(&mut canvas, &viscolors, &osc_colors, &ys);        
        //println!("{ys:?}");

        canvas.present();


        std::thread::sleep(std::time::Duration::from_millis(16));
    }

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
