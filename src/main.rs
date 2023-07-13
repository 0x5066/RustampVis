//ChatGPT assisted me with this.
//Base code came from https://stackoverflow.com/questions/72181090/getting-pixels-of-many-different-colors-when-drawing-with-sdl-in-rust
extern crate sdl2;

use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::rect::Rect;
use std::f64::consts::PI;

const WINDOW_WIDTH: i32 = 75;
const WINDOW_HEIGHT: i32 = 16;

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("rust-sdl2 demo", 800, 600)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.set_draw_color(Color::RGB(255, 255, 255));

    let xs: Vec<i32> = (0..WINDOW_WIDTH).collect();
    let ys: Vec<i32> = (0..WINDOW_WIDTH)
    .map(|x| ((WINDOW_HEIGHT as f64 / 2.0) * (1.0 + (2.0 * PI * x as f64 / WINDOW_WIDTH as f64).sin())) as i32)
    .collect();

    let mut last_y = 0;

    for (x, y) in xs.iter().zip(ys.iter()) {
        let x = *x;
        let y = *y;

        if x == 0 {
            last_y = y;
        }

        let mut top = y.min(last_y);
        let mut bottom = y.max(last_y);
        last_y = y;

        if bottom < top {
            let temp = bottom;
            bottom = top;
            top = temp;
            top += 1;
        }

        for dy in top..=bottom {
            let point = Point::new(x, dy);
            canvas.draw_point(point).unwrap();
        }
    }

    //canvas.fill_rect(Rect::new(10, 10, 80, 70)).unwrap();
    canvas.present();

    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }
    }
}
