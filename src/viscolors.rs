use std::fs::File;
use std::io::{BufRead, BufReader};
use sdl2::pixels::Color;
use std::path::Path;
use image::{open};

pub fn load_colors(file_path: &str) -> Vec<Color> {
    let mut colors = Vec::new();
    let path = Path::new(file_path);
    let extension = path.extension().and_then(|ext| ext.to_str());

    match extension {
        Some("txt") => {
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
        }
        Some("png") | Some("jpg") | Some("jpeg") | Some("gif") | Some("bmp") => {
            // Load colors from an image file
            if let Ok(img) = open(file_path) {
                for pixel in img.to_rgba8().pixels() {
                    let pixel = pixel;
                    let color = Color::RGB(pixel[0], pixel[1], pixel[2]);
                    colors.push(color);
                }
            } else {
                // Handle errors when opening the image
                eprintln!("Failed to open the image file");
            }
        }
        _ => {
            // Handle unsupported file types or other cases
            eprintln!("Unsupported file type: {:?}", extension);
        }
    }

    colors
}
