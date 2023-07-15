extern crate sdl2;

use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use std::f64::consts::PI;

mod viscolors;

const WINDOW_WIDTH: i32 = 75;
const WINDOW_HEIGHT: i32 = 16;
const SCROLL_SPEED: f64 = 0.75;
const ZOOM: i32 = 5;

fn draw_oscilloscope(canvas: &mut sdl2::render::Canvas<sdl2::video::Window>, scroll: f64, _colors: &[Color], osc_colors: &[Color]) {
    let xs: Vec<i32> = (0..WINDOW_WIDTH).collect();
    let ys: Vec<i32> = (0..WINDOW_WIDTH)
        .map(|x| ((WINDOW_HEIGHT as f64 / 2.0) * (1.0 + (2.0 * PI * (x as f64 + scroll) / WINDOW_WIDTH as f64).sin())) as i32)
        .collect();

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

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("rust-sdl2 demo", 400, 400)
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
    let mut scroll: f64 = 0.0;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        draw_oscilloscope(&mut canvas, scroll, &viscolors, &osc_colors);
        canvas.present();

        scroll += SCROLL_SPEED;

        std::thread::sleep(std::time::Duration::from_millis(16));
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
