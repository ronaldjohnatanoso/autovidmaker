#version 330
uniform sampler2D tex;           
uniform sampler2D subtitle_tex;  
uniform float u_time;            
uniform vec2 subtitle_size;      
uniform vec2 subtitle_pos;       
uniform float show_subtitle;     
in vec2 uv;                      
out vec4 color;                  

// ============================================================================
// HARDCORE RETENTION EFFECTS
// ============================================================================

/**
 * AGGRESSIVE CHROMATIC ABERRATION - Makes colors "explode" at edges
 */
vec3 effect_chromatic_aberration(sampler2D tex, vec2 uv_coords, float time) {
    float intensity = 0.015 + sin(time * 2.0) * 0.01; // Pulsating intensity
    
    vec2 center = vec2(0.5);
    vec2 direction = normalize(uv_coords - center);
    float distance = length(uv_coords - center);
    
    // Sample RGB channels at different offsets
    float r = texture(tex, uv_coords + direction * intensity * distance).r;
    float g = texture(tex, uv_coords).g;
    float b = texture(tex, uv_coords - direction * intensity * distance).b;
    
    return vec3(r, g, b);
}

/**
 * HYPNOTIC SPIRAL DISTORTION - Creates mesmerizing spiral effect
 */
vec2 effect_hypnotic_spiral(vec2 input_uv, float time) {
    vec2 center = vec2(0.5);
    vec2 centered_uv = input_uv - center;
    
    float radius = length(centered_uv);
    float angle = atan(centered_uv.y, centered_uv.x);

    // Only apply spiral to outer 50% (radius > 0.250)
    float spiral_threshold = 0.250;
    float spiral_intensity = 3.0;
    float spiral_speed = 1.5;
    
    if (radius > spiral_threshold) {
        // Create smooth transition from no effect to full effect
        float effect_strength = (radius - spiral_threshold) / (1.0 - spiral_threshold);
        effect_strength = smoothstep(0.0, 1.0, effect_strength); // Smooth transition
        
        angle += radius * spiral_intensity * sin(time * spiral_speed) * effect_strength;
    }
    
    // Apply slight zoom pulse only to outer area as well
    if (radius > spiral_threshold) {
        float pulse = 1.0 + sin(time * 2.0) * 0.05;
        float effect_strength = (radius - spiral_threshold) / (1.0 - spiral_threshold);
        effect_strength = smoothstep(0.0, 1.0, effect_strength);
        radius *= mix(1.0, pulse, effect_strength);
    }
    
    return center + vec2(cos(angle), sin(angle)) * radius;
}

/**
 * GLITCH EFFECT - Random digital corruption for attention grabbing
 */
vec3 effect_glitch(vec3 input_color, vec2 uv_coords, float time) {
    // Random horizontal lines
    float line_noise = sin(uv_coords.y * 1000.0 + time * 50.0);
    float line_effect = step(0.98, line_noise);
    
    // Color channel shifting
    float shift_intensity = line_effect * 0.1;
    vec3 shifted_color = input_color;
    shifted_color.r = mix(input_color.r, input_color.g, shift_intensity);
    shifted_color.b = mix(input_color.b, input_color.r, shift_intensity);
    
    // Random brightness spikes - now grey instead of white
    float brightness_noise = sin(time * 30.0 + uv_coords.x * 500.0);
    float brightness_spike = step(0.99, brightness_noise) * 0.5; // Reduced from 2.0 to 0.5 and made grey
    
    return shifted_color + vec3(brightness_spike * 0.5); // Make it grey by multiplying by 0.5
}

/**
 * AGGRESSIVE VIGNETTE WITH PULSING
 */
vec3 effect_hardcore_vignette(vec3 input_color, vec2 uv_coords, float time) {
    vec2 vignette_center = uv_coords - 0.5;
    
    // Pulsating vignette strength - reduced values for smaller reach
    float pulse = sin(time * 3.0) * 0.5 + 0.5;
    float vignette_strength = 1.1 + pulse * 0.6; // 1.2 to 1.8 intensity (reduced from 2.0-4.0)
    
    float vignette_factor = 1.0 - dot(vignette_center, vignette_center) * vignette_strength;
    vignette_factor = max(vignette_factor, 0.1); // Prevent complete blackout
    
    return input_color * vignette_factor;
}

/**
 * EXTREME SCANLINES WITH INTERFERENCE
 */
vec3 effect_hardcore_scanlines(vec3 input_color, vec2 uv_coords, float time) {
    float scanline_count = 150.0;
    float scroll_speed = 40.0;
    
    // Moving scanlines
    float line_position = uv_coords.y * scanline_count + time * scroll_speed;
    float line_cycle = fract(line_position);
    
    // Make scanlines more dramatic
    float scanline_width = 0.7;
    float scanline = step(scanline_width, line_cycle);
    
    // Very dark scanlines
    float min_brightness = 0.1;
    scanline = mix(min_brightness, 1.0, scanline);
    
    // Add aggressive interference
    float interference = 1.0 + sin(uv_coords.x * 2000.0 + time * 100.0) * 0.02;
    float vertical_interference = 1.0 + sin(uv_coords.y * 500.0 + time * 80.0) * 0.01;
    
    return input_color * scanline * interference * vertical_interference;
}

/**
 * ATTENTION-GRABBING COLOR CYCLING
 */
vec3 effect_color_cycling(vec3 input_color, float time) {
    // Aggressive color temperature shifts
    float cycle_speed = 2.0;
    float temp_shift = sin(time * cycle_speed) * 0.3; // Much stronger than before
    
    // Saturation boost during shifts
    float sat_boost = 1.3 + abs(temp_shift) * 0.5;
    
    // Apply dramatic temperature shift
    vec3 warm_shift = input_color * vec3(1.0 + temp_shift, 1.0, 1.0 - temp_shift * 0.7);
    
    // Boost saturation aggressively
    float luminance = dot(warm_shift, vec3(0.299, 0.587, 0.114));
    vec3 saturated = mix(vec3(luminance), warm_shift, sat_boost);
    
    // Removed color inversion pulse to eliminate white flashes
    return saturated;
}

/**
 * AGGRESSIVE KEN BURNS WITH SHAKE
 */
vec2 effect_hardcore_ken_burns(vec2 input_uv, float time) {
    float phi = 1.618033988749895;
    
    // Faster, more aggressive movement
    float zoom_speed = 0.15;
    float pan_speed = 0.08;
    float max_zoom = 1.6;
    
    // Add camera shake
    float shake_intensity = 0.003;
    vec2 shake = vec2(
        sin(time * 50.0) * shake_intensity,
        cos(time * 37.0) * shake_intensity
    );
    
    float zoom_factor = 1.0 + (sin(time * zoom_speed) * 0.5 + 0.5) * (max_zoom - 1.0);
    
    vec2 spiral_offset = vec2(
        sin(time * pan_speed * phi) * 0.12,
        cos(time * pan_speed / phi) * 0.08
    );
    
    vec2 center = vec2(0.5, 0.5) + spiral_offset + shake;
    return center + (input_uv - center) / zoom_factor;
}

/**
 * BARREL DISTORTION WITH WOBBLE
 */
vec2 effect_wobble_distortion(vec2 input_uv, float time) {
    vec2 centered_uv = input_uv - 0.5;
    
    // Wobbling distortion strength
    float wobble = sin(time * 3.0) * 0.2 + 0.4; // 0.2 to 0.6
    
    float r2 = dot(centered_uv, centered_uv);
    float distortion_factor = 1.0 + wobble * r2;
    
    vec2 distorted_uv = centered_uv * distortion_factor + 0.5;
    return distorted_uv;
}

// ============================================================================
// SUBTITLE AND MAIN FUNCTION
// ============================================================================

vec3 add_subtitle_overlay(vec3 base_color, vec2 uv_coords) {
    if (show_subtitle < 0.5 || subtitle_size.x <= 0.0 || subtitle_size.y <= 0.0) {
        return base_color;
    }
    
    vec2 subtitle_uv = (uv_coords - subtitle_pos) / subtitle_size;
    
    if (subtitle_uv.x >= 0.0 && subtitle_uv.x <= 1.0 && subtitle_uv.y >= 0.0 && subtitle_uv.y <= 1.0) {
        vec4 subtitle_sample = texture(subtitle_tex, subtitle_uv);
        return mix(base_color, subtitle_sample.rgb, subtitle_sample.a);
    }
    
    return base_color;
}

void main() {
    vec2 working_uv = uv;
    
    // Apply HARDCORE transformations
    working_uv = effect_hardcore_ken_burns(working_uv, u_time);
    working_uv = effect_hypnotic_spiral(working_uv, u_time);
    working_uv = effect_wobble_distortion(working_uv, u_time);
    
    // Bounds check with more lenient limits
    if (working_uv.x < -0.2 || working_uv.x > 1.2 || working_uv.y < -0.2 || working_uv.y > 1.2) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    vec2 clamped_uv = clamp(working_uv, 0.0, 1.0);
    
    // Sample with chromatic aberration
    vec3 working_color = effect_chromatic_aberration(tex, clamped_uv, u_time);
    
    // Apply HARDCORE color effects
    working_color = effect_glitch(working_color, uv, u_time);
    working_color = effect_color_cycling(working_color, u_time);
    // working_color = effect_hardcore_scanlines(working_color, uv, u_time); // DISABLED
    working_color = effect_hardcore_vignette(working_color, uv, u_time);
    
    // Subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}