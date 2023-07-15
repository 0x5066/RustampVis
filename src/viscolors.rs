use std::fs::File;
use std::io::{BufRead, BufReader};
use sdl2::pixels::Color;

pub fn load_colors(file_path: &str) -> Vec<Color> {
    let mut colors = Vec::new();
    let file = File::open(file_path).expect("Failed to open file");

    for line in BufReader::new(file).lines() {
        if let Ok(line) = line {
            if line.starts_with("//") {
                continue;
            }

            let values: Vec<u8> = line
                .split(',')
                .take(3)
                .map(|value| value.trim().parse().unwrap_or(0))
                .collect();

            if values.len() == 3 {
                let color = Color::RGB(values[0], values[1], values[2]);
                colors.push(color);
            }
        }
    }

    colors
}
