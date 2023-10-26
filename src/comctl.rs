extern crate sdl2;

use sdl2::ttf::Font;
use sdl2::rect::Rect;
use sdl2::pixels::Color;
use sdl2::video::WindowContext;
use sdl2::render::TextureCreator;
use sdl2::rect::Point;

pub fn newline_handler<'a>(
    text: &'a str,
    font: &'a Font<'a, 'a>,
    texture_creator: &'a TextureCreator<sdl2::video::WindowContext>,
    color: Color,
) -> Result<Vec<sdl2::render::Texture<'a>>, String> {
    // Split the input text into lines based on '\n'
    let lines: Vec<&str> = text.split('\n').collect();

    // Create textures for each line
    let mut textures = Vec::new();
    for line in lines {
        let surface = font.render(line).solid(color).map_err(|e| e.to_string())?;
        let texture = texture_creator.create_texture_from_surface(&surface).map_err(|e| e.to_string())?;
        textures.push(texture);
    }

    Ok(textures)
}

pub fn listview_box(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    // Fill the background with a color
    let rect = Rect::new(x, y, width, height);
    canvas.set_draw_color(cgenex[0]);
    canvas.fill_rect(rect)?;

    // Draw the bottom and right lines
    canvas.set_draw_color(cgenex[5]);

    // Bottom line
    canvas.draw_line(Point::new(x, y + height as i32), Point::new(x + width as i32, y + height as i32))?;

    // Right line
    canvas.draw_line(Point::new(x + width as i32, y + height as i32), Point::new(x + width as i32, y))?;

    // Draw the darker lines slightly above the bottom and to the left of the right line
    canvas.set_draw_color(cgenex[10]);

    // Line slightly above the bottom
    canvas.draw_line(Point::new(x, y + height as i32 - 1), Point::new(x + width as i32 - 1, y + height as i32 - 1))?;

    // Line to the left of the right line
    canvas.draw_line(Point::new(x + width as i32 - 1, y + height as i32 - 1), Point::new(x + width as i32 - 1, y))?;

    Ok(())
}

pub fn render_text(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    font: &Font,
    text: &str,
    color: Color,
    target_x: i32,
    target_y: i32,
    texture_creator: &TextureCreator<WindowContext>,
) -> Result<(), String> {
    let surface = font.render(text).solid(color).map_err(|e| e.to_string())?;
    let texture = texture_creator.create_texture_from_surface(&surface).map_err(|e| e.to_string())?;
    let (text_width, text_height) = font.size_of(text).map_err(|e| e.to_string())?;
    let target = Rect::new(target_x, target_y, text_width, text_height);
    canvas.copy(&texture, None, Some(target))?;
    Ok(())
}

pub fn groupbox(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    text: &str,
    font: &sdl2::ttf::Font,
    texture_creator: &TextureCreator<WindowContext>,
    color: &[Color],
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    canvas.set_draw_color(color[2]);

    // Calculate the text dimensions
    let (text_width, text_height) = font.size_of(text).map_err(|e| e.to_string())?;

    // Create texture for the text
    let surface = font.render(text).solid(color[1]).map_err(|e| e.to_string())?;
    let texture = texture_creator.create_texture_from_surface(&surface).map_err(|e| e.to_string())?;
    let target = Rect::new(x + 9, y, text_width, text_height);
    canvas.copy(&texture, None, Some(target))?;

    // Draw horizontal and vertical lines inside the groupbox
    canvas.set_draw_color(color[10]); // Set line color
    // Top horizontal line
    canvas.draw_line(Point::new(x, y+5), Point::new(x + 7 as i32, y+5)).unwrap();
    canvas.draw_line(Point::new(x + text_width as i32 + 15, y+5), Point::new(x + width as i32, y+5)).unwrap();

    // Left vertical line
    canvas.draw_line(Point::new(x, y+5), Point::new(x, y + height as i32)).unwrap();

    // Right vertical line
    canvas.draw_line(Point::new(x + width as i32, y+5), Point::new(x + width as i32, y + height as i32)).unwrap();

    // Bottom horizontal line
    canvas.draw_line(Point::new(x, y + height as i32), Point::new(x + width as i32, y + height as i32)).unwrap();

    // indented top line
    canvas.set_draw_color(color[5]);
    canvas.draw_line(Point::new(x + 1, y+6), Point::new(x + 8 as i32, y + 6)).unwrap();
    canvas.draw_line(Point::new(x + text_width as i32 + 14, y+6), Point::new(x + width as i32 - 1, y+6)).unwrap();

    // indented left line
    canvas.draw_line(Point::new(x + 1, y+6), Point::new(x + 1, y + height as i32 - 1)).unwrap();

    // indented bottom line
    canvas.draw_line(Point::new(x, y + height as i32 + 1), Point::new(x + width as i32 + 1, y + height as i32 + 1)).unwrap();

    // Right vertical line
    canvas.draw_line(Point::new(x + width as i32 + 1, y+5), Point::new(x + width as i32 + 1, y + height as i32)).unwrap();

    Ok(())
}

pub fn draw_dropdown(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    // Draw the border
    canvas.set_draw_color(cgenex[5]);
    let border_rect = Rect::new(x, y, width, height);
    canvas.draw_rect(border_rect)?;

    // Fill the dropdown box
    canvas.set_draw_color(cgenex[0]);
    let fill_rect = Rect::new(x + 1, y + 1, width - 2, height - 2);
    canvas.fill_rect(fill_rect)?;

    Ok(())
}

pub fn checkbox(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    x: i32,
    y: i32,
    text: &str,
    font: &sdl2::ttf::Font,
    marlett: &sdl2::ttf::Font,
    texture_creator: &TextureCreator<WindowContext>,
) -> Result<(), String> {
    canvas.set_draw_color(cgenex[5]);
    let border_rect = Rect::new(x, y, 13, 13);
    canvas.draw_rect(border_rect)?;

    canvas.set_draw_color(cgenex[0]);
    let fill_rect = Rect::new(x + 1, y + 1, 11, 11);
    canvas.fill_rect(fill_rect)?;

    render_text(canvas, font, text, cgenex[4], x + 17, y, texture_creator)?;

    render_text(canvas, marlett, "b", cgenex[1], x - 1, y, texture_creator)?;

    Ok(())
}