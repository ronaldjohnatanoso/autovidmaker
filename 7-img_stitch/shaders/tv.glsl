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
// HYPNOTIC FRACTAL EFFECTS
// ============================================================================

/**
 * MANDELBROT-INSPIRED FRACTAL OVERLAY
 */
vec3 effect_fractal_overlay(vec3 base_color, vec2 uv_coords, float time) {
    vec2 c = (uv_coords - 0.5) * 3.0; // Scale and center
    c += vec2(cos(time * 0.3), sin(time * 0.2)) * 0.5; // Slow drift
    
    vec2 z = vec2(0.0);
    float iterations = 0.0;
    const float max_iter = 32.0;
    
    for (float i = 0.0; i < max_iter; i++) {
        if (length(z) > 2.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }
    
    float fractal_value = iterations / max_iter;
    
    // Create shifting color palette
    vec3 fractal_color = vec3(
        sin(fractal_value * 6.28 + time) * 0.5 + 0.5,
        sin(fractal_value * 6.28 + time + 2.09) * 0.5 + 0.5,
        sin(fractal_value * 6.28 + time + 4.18) * 0.5 + 0.5
    );
    
    // Blend with base image
    float blend_factor = 0.3 + sin(time * 2.0) * 0.2;
    return mix(base_color, fractal_color, blend_factor);
}

/**
 * INFINITE ZOOM SPIRAL
 */
vec2 effect_infinite_spiral(vec2 input_uv, float time) {
    vec2 center = vec2(0.5);
    vec2 centered_uv = input_uv - center;
    
    float radius = length(centered_uv);
    float angle = atan(centered_uv.y, centered_uv.x);
    
    // Infinite zoom with spiral
    float zoom_factor = mod(time * 0.5, 4.0); // Cycle every 8 seconds
    radius *= pow(2.0, zoom_factor);
    
    // Spiral rotation
    angle += log(radius) * 0.5 + time * 0.8;
    
    return center + vec2(cos(angle), sin(angle)) * fract(radius);
}

/**
 * HYPNOTIC KALEIDOSCOPE
 */
vec2 effect_kaleidoscope(vec2 input_uv, float time) {
    vec2 center = vec2(0.5);
    vec2 centered_uv = input_uv - center;
    
    float radius = length(centered_uv);
    float angle = atan(centered_uv.y, centered_uv.x);
    
    // Create kaleidoscope segments
    float segments = 6.0;
    angle = mod(angle + time * 0.5, 6.28318 / segments);
    if (mod(floor(angle * segments / 6.28318), 2.0) == 1.0) {
        angle = 6.28318 / segments - angle;
    }
    
    return center + vec2(cos(angle), sin(angle)) * radius;
}

/**
 * MORPHING JULIA SET
 */
vec3 effect_julia_morph(vec3 base_color, vec2 uv_coords, float time) {
    vec2 z = (uv_coords - 0.5) * 2.5;
    
    // Morphing Julia set parameter
    vec2 c = vec2(
        cos(time * 0.3) * 0.7,
        sin(time * 0.2) * 0.7
    );
    
    float iterations = 0.0;
    const float max_iter = 20.0;
    
    for (float i = 0.0; i < max_iter; i++) {
        if (length(z) > 2.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }
    
    float julia_value = iterations / max_iter;
    
    // Psychedelic color cycling
    vec3 julia_color = vec3(
        sin(julia_value * 10.0 + time * 3.0) * 0.5 + 0.5,
        sin(julia_value * 10.0 + time * 3.0 + 2.0) * 0.5 + 0.5,
        sin(julia_value * 10.0 + time * 3.0 + 4.0) * 0.5 + 0.5
    );
    
    return mix(base_color, julia_color, 0.4);
}

/**
 * SIERPINSKI TRIANGLE OVERLAY
 */
vec3 effect_sierpinski(vec3 base_color, vec2 uv_coords, float time) {
    vec2 coord = uv_coords * 8.0 + time * 0.5; // Scale and animate
    
    float sierpinski = 1.0;
    for (int i = 0; i < 8; i++) {
        coord *= 2.0;
        vec2 fractal_coord = floor(coord);
        if (fractal_coord.x + fractal_coord.y == 1.0) {
            sierpinski = 0.0;
            break;
        }
        coord = fract(coord);
    }
    
    vec3 triangle_color = vec3(
        sin(time * 2.0) * 0.5 + 0.5,
        sin(time * 2.0 + 2.0) * 0.5 + 0.5,
        sin(time * 2.0 + 4.0) * 0.5 + 0.5
    );
    
    return mix(base_color, triangle_color, sierpinski * 0.3);
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
    
    // Apply hypnotic transformations
    working_uv = effect_infinite_spiral(working_uv, u_time);
    working_uv = effect_kaleidoscope(working_uv, u_time);
    
    // Sample the texture
    vec3 working_color = texture(tex, working_uv).rgb;
    
    // Apply fractal overlays
    working_color = effect_fractal_overlay(working_color, uv, u_time);
    working_color = effect_julia_morph(working_color, uv, u_time);
    working_color = effect_sierpinski(working_color, uv, u_time);
    
    // Subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}