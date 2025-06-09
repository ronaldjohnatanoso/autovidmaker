#version 330
uniform sampler2D tex;           // The image texture
uniform sampler2D subtitle_tex;  // The subtitle texture
uniform float u_time;            // Time for animations
uniform vec2 subtitle_size;      // Size of the subtitle texture
uniform vec2 subtitle_pos;       // Position of the subtitle
uniform float show_subtitle;     // Whether to show subtitle (0.0 or 1.0)
in vec2 uv;                      // UV coordinates from vertex shader
out vec4 color;                  // Final pixel color

// ============================================================================
// INDEPENDENT EFFECT MODULES
// ============================================================================

/**
 * ZOOM MODULE - Completely self-contained
 * Input: any UV coordinates
 * Output: zoomed UV coordinates
 */
vec2 effect_zoom(vec2 input_uv, float time) {
    vec2 center = vec2(0.5, 0.5);
    float zoom_factor = 1.0 + sin(time * 2.0) * 0.3; // speed=2.0, intensity=0.3
    return center + (input_uv - center) / zoom_factor;
}

/**
 * SCANLINE NOISE MODULE - Completely independent color effect
 * Input: base color
 * Output: color with noise added
 */
vec3 effect_scanlines(vec3 input_color, vec2 uv_coords, float time) {
    float noise = sin(uv_coords.y * 1200.0 + time * 10.0) * 0.02;
    return input_color + noise;
}

/**
 * VIGNETTE MODULE - Completely independent color effect
 * Input: base color
 * Output: color with vignette applied
 */
vec3 effect_vignette(vec3 input_color, vec2 uv_coords) {
    vec2 vignette_center = uv_coords - 0.5;
    float vignette_factor = 1.0 - dot(vignette_center, vignette_center) * 0.9;
    return input_color * vignette_factor;
}

/**
 * DESATURATION MODULE - Completely independent color effect
 * Input: color
 * Output: desaturated color
 */
vec3 effect_desaturate(vec3 input_color, float amount) {
    float gray = dot(input_color, vec3(0.299, 0.587, 0.114));
    return mix(input_color, vec3(gray), amount);
}

/**
 * SUBTITLE OVERLAY MODULE - Renders subtitles with alpha blending
 */
vec3 add_subtitle_overlay(vec3 base_color, vec2 uv_coords) {
    if (show_subtitle < 0.5 || subtitle_size.x <= 0.0 || subtitle_size.y <= 0.0) {
        return base_color;
    }
    
    // Calculate UV coordinates for the subtitle
    vec2 subtitle_uv = (uv_coords - subtitle_pos) / subtitle_size;
    
    // Check if we're within the subtitle area
    if (subtitle_uv.x >= 0.0 && subtitle_uv.x <= 1.0 && subtitle_uv.y >= 0.0 && subtitle_uv.y <= 1.0) {
        vec4 subtitle_sample = texture(subtitle_tex, subtitle_uv);
        
        // Blend subtitle over the base image using subtitle alpha
        return mix(base_color, subtitle_sample.rgb, subtitle_sample.a);
    }
    
    return base_color;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
void main() {
    // Start with original UV coordinates
    vec2 working_uv = uv;
    
    // Apply UV transformations (comment to disable)
    working_uv = effect_zoom(working_uv, u_time);
    
    // Sample texture with final UV coordinates
    vec4 base_texture = texture(tex, working_uv);
    vec3 working_color = base_texture.rgb;
    
    // Apply color effects (comment to disable)
    working_color = effect_scanlines(working_color, uv, u_time);
    working_color = effect_vignette(working_color, uv);
    working_color = effect_desaturate(working_color, 0.1);
    
    // Add subtitle overlay
    working_color = add_subtitle_overlay(working_color, uv);
    
    // Output final result
    color = vec4(working_color, base_texture.a);
}