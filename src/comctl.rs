extern crate sdl2;

use sdl2::ttf::Font;
use sdl2::rect::Rect;
use sdl2::pixels::Color;
use sdl2::video::WindowContext;
use sdl2::render::TextureCreator;
use sdl2::rect::Point;
use sdl2::image::LoadTexture;

use std::sync::{Arc, Mutex};

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
    canvas.set_draw_color(cgenex[9]);

    // Bottom line
    canvas.draw_line(Point::new(x, y + height as i32), Point::new(x + width as i32, y + height as i32))?;

    // Right line
    canvas.draw_line(Point::new(x + width as i32, y + height as i32), Point::new(x + width as i32, y))?;

    // Draw the darker lines slightly above the bottom and to the left of the right line
    canvas.set_draw_color(cgenex[7]);

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
    canvas.draw_line(Point::new(x, y+5), Point::new(x + 8 as i32, y+5)).unwrap();
    canvas.draw_line(Point::new(x + text_width as i32 + 14, y+5), Point::new(x + width as i32, y+5)).unwrap();

    // Left vertical line
    canvas.draw_line(Point::new(x, y+5), Point::new(x, y + height as i32)).unwrap();

    // Right vertical line
    canvas.draw_line(Point::new(x + width as i32, y+5), Point::new(x + width as i32, y + height as i32)).unwrap();

    // Bottom horizontal line
    canvas.draw_line(Point::new(x, y + height as i32), Point::new(x + width as i32, y + height as i32)).unwrap();

    // indented top line
    canvas.set_draw_color(color[9]);
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
    item: Arc<Mutex<u8>>,
    is_button_clicked: bool,
    mx: i32,
    my: i32,
) -> Result<(), String> {
    let mut item_value = item.lock().unwrap();
    canvas.set_draw_color(cgenex[5]);
    let border_rect = Rect::new(x, y, 13, 13);
    canvas.draw_rect(border_rect)?;

    canvas.set_draw_color(cgenex[0]);
    let fill_rect = Rect::new(x + 1, y + 1, 11, 11);
    canvas.fill_rect(fill_rect)?;

    render_text(canvas, font, text, cgenex[4], x + 17, y, texture_creator)?;


    if mx >= x && mx <= x + 13 as i32 && my >= y && my <= y + 13 as i32 && is_button_clicked {
        // Toggle item_value between 0 and 1
        *item_value = 1 - *item_value;
    }

    // Render text based on the updated item_value
    if *item_value == 1 {
        render_text(canvas, marlett, "b", cgenex[1], x - 1, y, texture_creator)?;
    }

    Ok(())
}

pub fn slider_small(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    text: &str,
    font: &sdl2::ttf::Font,
    texture_creator: &TextureCreator<WindowContext>,
    max_value: i32,
) -> Result<(), String> {

    render_text(canvas, font, text, cgenex[4], x, y, texture_creator)?;

    canvas.set_draw_color(cgenex[10]);
    canvas.draw_line(Point::new(x+2, y+25), Point::new(x + width as i32 - 2, y+25)).unwrap();
    canvas.draw_line(Point::new(x+2, y+25), Point::new(x + 2, y+27)).unwrap();

    canvas.set_draw_color(cgenex[11]);
    canvas.draw_line(Point::new(x+3, y+26), Point::new(x + width as i32 - 3, y+26)).unwrap();

    canvas.set_draw_color(cgenex[9]);
    canvas.draw_line(Point::new(x+2, y+28), Point::new(x + width as i32 - 2, y+28)).unwrap();
    canvas.draw_line(Point::new(x + width as i32 - 1, y+25), Point::new(x + width as i32 - 1, y+28)).unwrap();

    canvas.set_draw_color(cgenex[7]);
    canvas.draw_line(Point::new(x+3, y+27), Point::new(x + width as i32 - 3, y+27)).unwrap();
    canvas.draw_line(Point::new(x + width as i32 - 2, y+27), Point::new(x + width as i32 - 2, y+26)).unwrap();
    //canvas.draw_line(Point::new(x+2, y+25), Point::new(x + 2, y+27)).unwrap();

    for i in 0..=max_value {
        let line_x = x + (i as f32 / max_value as f32 * (width - 6) as f32) as i32 + 3;
        canvas.set_draw_color(cgenex[4]);
        canvas.draw_line(Point::new(line_x, y + 34), Point::new(line_x, y + 36)).unwrap();
    }
    
    Ok(())
}


pub fn button(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    text: &str,
    font: &sdl2::ttf::Font,
    texture_creator: &TextureCreator<WindowContext>,
    image_path: &str,
    is_button_clicked: bool,
    mx: i32,
    my: i32,
) -> Result<(), String> {

    // Calculate the text dimensions
    let (text_width, text_height) = font.size_of(text).map_err(|e| e.to_string())?;

    // Create texture for the text
    let surface = font.render(text).solid(cgenex[3]).map_err(|e| e.to_string())?;
    let texture = texture_creator.create_texture_from_surface(&surface).map_err(|e| e.to_string())?;

    // Calculate the x and y positions to center the text within the button
    let text_x = x + (width as i32 - text_width as i32) / 2;
    let text_y = y + (height as i32 - text_height as i32) / 2;

    // Create a Rect for the text
    let target;

    let image_texture = texture_creator.load_texture(image_path)?;

/*     canvas.set_draw_color(cgenex[9]);
    canvas.draw_line(Point::new(x, y), Point::new(x + width as i32, y)).unwrap();
    canvas.draw_line(Point::new(x, y), Point::new(x as i32, y + height as i32 - 1)).unwrap();

    canvas.set_draw_color(cgenex[10]);
    canvas.draw_line(Point::new(x + 1, y + height as i32 - 1), Point::new(x + width as i32 , y + height as i32 - 1)).unwrap();
    canvas.draw_line(Point::new(x + width as i32 - 1, y + 1), Point::new(x + width as i32 - 1, y + height as i32 - 1)).unwrap();

    canvas.set_draw_color(cgenex[11]);
    canvas.draw_line(Point::new(x, y + height as i32), Point::new(x + width as i32 , y + height as i32)).unwrap();
    canvas.draw_line(Point::new(x + width as i32, y), Point::new(x + width as i32, y + height as i32)).unwrap();

    let rect = Rect::new(x+1, y+1, width-2, height-2);
    canvas.set_draw_color(cgenex[7]);
    canvas.fill_rect(rect)?; */

        // Define the source and destination rectangles to copy a piece of the image
        let text_offset: i32;
        let button_offset: i32;

        let src_tl_rect;
        let dst_tl_rect;

        let src_tm_rect;
        let dst_tm_rect;

        let src_tr_rect;
        let dst_tr_rect;

        let src_l_rect;
        let dst_l_rect;

        let src_m_rect;
        let dst_m_rect;

        let src_r_rect;
        let dst_r_rect;

        let src_bl_rect;
        let dst_bl_rect;

        let src_bm_rect;
        let dst_bm_rect;

        let src_br_rect;
        let dst_br_rect;

        if mx >= x && mx <= x + width as i32 && my >= y && my <= y + height as i32 && is_button_clicked {
            text_offset = 2;
            button_offset = 15;
        } else {
            text_offset = 0;
            button_offset = 0;
        }

        src_tl_rect = Rect::new(0, 0 + button_offset, 4, 4); // src image
        dst_tl_rect = Rect::new(x, y, 4, 4); // dest image

        src_tm_rect = Rect::new(4, 0 + button_offset, 39, 4);
        dst_tm_rect = Rect::new(x + 4, y, width - 7, 4);

        src_tr_rect = Rect::new(43, 0 + button_offset, 4, 4);
        dst_tr_rect = Rect::new(x + width as i32 - 3, y, 4, 4);

        src_l_rect = Rect::new(0, 4 + button_offset, 4, 7);
        dst_l_rect = Rect::new(x, y + 4, 4, height - 7);

        src_m_rect = Rect::new(4, 4 + button_offset, 39, 7);
        dst_m_rect = Rect::new(x + 4, y + 4, width - 7, height - 7);

        src_r_rect = Rect::new(43, 4 + button_offset, 4, 7);
        dst_r_rect = Rect::new(x + width as i32 - 3, y + 4, 4, height - 7);

        src_bl_rect = Rect::new(0, 11 + button_offset, 4, 4);
        dst_bl_rect = Rect::new(x, y + height as i32 - 3, 4, 4);

        src_bm_rect = Rect::new(4, 11 + button_offset, 39, 4);
        dst_bm_rect = Rect::new(x + 4, y + height as i32 - 3, width - 7, 4);

        src_br_rect = Rect::new(43, 11 + button_offset, 4, 4);
        dst_br_rect = Rect::new(x + width as i32 - 3, y + height as i32 - 3, 4, 4);

        target = Rect::new(text_x + text_offset, text_y + text_offset, text_width, text_height);

        // Copy the specified part of the image to the canvas
        canvas.copy(&image_texture, src_tl_rect, dst_tl_rect)?;
        canvas.copy(&image_texture, src_tm_rect, dst_tm_rect)?;
        canvas.copy(&image_texture, src_tr_rect, dst_tr_rect)?;
        canvas.copy(&image_texture, src_l_rect, dst_l_rect)?;
        canvas.copy(&image_texture, src_m_rect, dst_m_rect)?;
        canvas.copy(&image_texture, src_r_rect, dst_r_rect)?;
        canvas.copy(&image_texture, src_bl_rect, dst_bl_rect)?;
        canvas.copy(&image_texture, src_bm_rect, dst_bm_rect)?;
        canvas.copy(&image_texture, src_br_rect, dst_br_rect)?;

    canvas.copy(&texture, None, Some(target))?;
    Ok(())
}

pub fn tab(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    x: i32,
    y: i32,
    text: &str,
    font: &sdl2::ttf::Font,
    texture_creator: &TextureCreator<WindowContext>,
) -> Result<(), String> {
    // Calculate the text dimensions
    let (text_width, text_height) = font.size_of(text).map_err(|e| e.to_string())?;

    // Create texture for the text
    let surface = font.render(text).solid(cgenex[1]).map_err(|e| e.to_string())?;
    let texture = texture_creator.create_texture_from_surface(&surface).map_err(|e| e.to_string())?;
    let target = Rect::new(x + 10, y + 3, text_width, text_height);

    canvas.set_draw_color(cgenex[9]);
    let rect = Rect::new(x, y, text_width + 20, text_height + 7);
    canvas.draw_rect(rect).unwrap();
    canvas.set_draw_color(cgenex[10]);
    let rect = Rect::new(x + 1, y + 1, text_width + 18, text_height + 5);
    canvas.fill_rect(rect).unwrap();

    canvas.copy(&texture, None, Some(target))?;

    Ok(())
}

pub fn radiobutton(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    cgenex: &[Color],
    x: i32,
    y: i32,
    options_str: &str,
    font: &sdl2::ttf::Font,
    marlett: &sdl2::ttf::Font,
    texture_creator: &TextureCreator<WindowContext>,
    selected_option: &str, // Pass a mutable reference to update the selected option
    is_button_clicked: bool,
    mx: i32,
    my: i32,
) -> Result<(), String> {
    let options: Vec<&str> = options_str.split(';').collect();
    //let mut selected_option_value = selected_option.lock().unwrap();
    let mut current_x = x + 10;

    for option in &options {

        //let radio_button_rect = Rect::new(current_x - 20, y, option_text_width, option_text_height);

        /*if radio_button_rect.contains_point(Point::new(mx, my)) && is_button_clicked {
            *selected_option_value = &option.to_string();
        }*/

        if selected_option == option.to_string().to_lowercase() {
            render_text(canvas, marlett, "n", cgenex[0], current_x - 20, y, texture_creator)?;
            render_text(canvas, marlett, "l", cgenex[5], current_x - 20, y, texture_creator)?;
            render_text(canvas, marlett, "m", cgenex[5], current_x - 20, y, texture_creator)?;
            render_text(canvas, marlett, "i", cgenex[1], current_x - 20, y, texture_creator)?;
        } else {
            render_text(canvas, marlett, "n", cgenex[0], current_x - 20, y, texture_creator)?;
            render_text(canvas, marlett, "l", cgenex[5], current_x - 20, y, texture_creator)?;
            render_text(canvas, marlett, "m", cgenex[5], current_x - 20, y, texture_creator)?;
        }

        // Render the option text
        render_text(canvas, font, option, cgenex[4], current_x, y, texture_creator)?;

        current_x += 90; // Adjust this value to control the spacing between radio buttons and text
    }

    Ok(())
}

