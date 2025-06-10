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
 * SUPER THICK SCANLINES - Maximum visibility with movement
 */
vec3 effect_scanlines(vec3 input_color, vec2 uv_coords, float time) {
    // Very thick scanlines
    float scanline_count = 180.0;           // Even fewer lines = even thicker
    float scroll_speed = 35.0;              // Speed of movement
    
    // Calculate which scanline we're on WITH movement
    float line_position = uv_coords.y * scanline_count + time * scroll_speed;
    float line_cycle = fract(line_position);
    
    // Create very thick dark bands
    float scanline_width = 0.6;            // 60% of each line is dark
    float scanline = step(scanline_width, line_cycle);
    
    // Make the dark areas very dark
    float min_brightness = 0.15;           // Dark lines are 15% brightness
    scanline = mix(min_brightness, 1.0, scanline);
    
    // Add slight random interference
    float interference = 1.0 + (sin(uv_coords.x * 1000.0 + time * 50.0) * 0.005);
    
    return input_color * scanline * interference;
}

/**
 * VIGNETTE MODULE - Completely independent color effect
 * Input: base color
 * Output: color with vignette applied
 */
vec3 effect_vignette(vec3 input_color, vec2 uv_coords) {
    vec2 vignette_center = uv_coords - 0.5;
    float vignette_factor = 1.0 - dot(vignette_center, vignette_center) *3.0 ; // Increased from 0.9 to 2.5
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

/**
 * CRT BARREL DISTORTION MODULE - Creates curved TV screen effect
 * Input: UV coordinates
 * Output: Distorted UV coordinates with barrel effect
 */
vec2 effect_crt_distortion(vec2 input_uv) {
    // Center the coordinates around 0
    vec2 centered_uv = input_uv - 0.5;
    
    // More aggressive barrel distortion
    float distortion_strength = 0.4;  // Increased from 0.15 to 0.4
    
    // Apply barrel distortion using a stronger formula
    float r2 = dot(centered_uv, centered_uv);
    float distortion_factor = 1.0 + distortion_strength * r2;
    
    vec2 distorted_uv = centered_uv * distortion_factor + 0.5;
    
    return distorted_uv;
}

/**
 * CRT EDGE FADE MODULE - Darkens edges like old TV with more aggressive fade
 */
vec3 effect_crt_edge_fade(vec3 input_color, vec2 uv_coords) {
    // More aggressive edge fade
    vec2 edge_distance = abs(uv_coords - 0.5);
    float max_edge_distance = max(edge_distance.x, edge_distance.y);
    
    // Make the fade start earlier and be more dramatic
    float edge_fade = 1.0 - smoothstep(0.35, 0.5, max_edge_distance);
    
    return input_color * edge_fade;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
void main() {
    // Start with original UV coordinates
    vec2 working_uv = uv;
    
    // Apply CRT barrel distortion FIRST (affects sampling)
    working_uv = effect_crt_distortion(working_uv);
    
    // Apply zoom after distortion (uncomment if you want zoom)
    // working_uv = effect_zoom(working_uv, u_time);
    
    // More lenient bounds check - allow some distortion outside normal range
    if (working_uv.x < -0.1 || working_uv.x > 1.1 || working_uv.y < -0.1 || working_uv.y > 1.1) {
        color = vec4(0.0, 0.0, 0.0, 1.0); // Black outside screen area
        return;
    }
    
    // Clamp UV to valid range for texture sampling
    vec2 clamped_uv = clamp(working_uv, 0.0, 1.0);
    
    // Sample texture with clamped UV coordinates
    vec4 base_texture = texture(tex, clamped_uv);
    vec3 working_color = base_texture.rgb;
    
    // Apply color effects (using original UV for effects, not distorted)
    working_color = effect_scanlines(working_color, uv, u_time);
    working_color = effect_vignette(working_color, uv);
    // working_color = effect_crt_edge_fade(working_color, uv); // CRT edge darkening
    // working_color = effect_desaturate(working_color, 0.2); // Increased desaturation
    
    // Add subtitle overlay (using original UV)
    working_color = add_subtitle_overlay(working_color, uv);
    
    // Output final result
    color = vec4(working_color, base_texture.a);
}