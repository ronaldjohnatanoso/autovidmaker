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
 * RETENTION-OPTIMIZED KEN BURNS - Uses golden ratio and Fibonacci spirals
 */
vec2 effect_ken_burns(vec2 input_uv, float time) {
    // Golden ratio-based movement (naturally pleasing to human eye)
    float phi = 1.618033988749895; // Golden ratio
    
    float zoom_speed = 0.08;       // Slower, more hypnotic
    float pan_speed = 0.03;        // Gentle drift
    float max_zoom = 1.4;          // Slightly more zoom
    
    // Fibonacci spiral-based zoom (mathematically pleasing)
    float zoom_factor = 1.0 + (sin(time * zoom_speed) * 0.5 + 0.5) * (max_zoom - 1.0);
    
    // Golden ratio spiral movement
    vec2 spiral_offset = vec2(
        sin(time * pan_speed * phi) * 0.08,
        cos(time * pan_speed / phi) * 0.05
    );
    
    vec2 center = vec2(0.5, 0.5) + spiral_offset;
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
    float vignette_factor = 1.0 - dot(vignette_center, vignette_center) * 3.0;
    return input_color * vignette_factor;
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
    float distortion_strength = 0.4;
    
    // Apply barrel distortion using a stronger formula
    float r2 = dot(centered_uv, centered_uv);
    float distortion_factor = 1.0 + distortion_strength * r2;
    
    vec2 distorted_uv = centered_uv * distortion_factor + 0.5;
    
    return distorted_uv;
}

/**
 * HYPNOTIC PULSE - Rhythmic zoom that creates subconscious engagement
 */
vec2 effect_hypnotic_pulse(vec2 input_uv, float time) {
    // Pulse parameters
    float pulse_speed = 1.5;       // Speed of pulse (heartbeat-like)
    float pulse_intensity = 0.08;  // How much it pulses
    
    // Create pulse based on heartbeat rhythm (60-80 BPM feels natural)
    float pulse = sin(time * pulse_speed) * 0.5 + 0.5;
    pulse = pow(pulse, 2.0); // Make it more dramatic
    
    // Apply subtle zoom pulse
    float zoom = 1.0 + pulse * pulse_intensity;
    vec2 center = vec2(0.5, 0.5);
    
    return center + (input_uv - center) / zoom;
}

/**
 * RETENTION BOOSTER - Subtle color temperature shifts
 */
vec3 effect_retention_boost(vec3 input_color, float time) {
    // Very subtle color temperature shifts (warmer/cooler)
    float temp_shift = sin(time * 0.3) * 0.1; // Slow, subtle shift
    
    // Slightly boost saturation during shifts
    float sat_boost = 1.0 + abs(temp_shift) * 0.2;
    
    // Apply temperature shift
    vec3 warm_shift = input_color * vec3(1.0 + temp_shift, 1.0, 1.0 - temp_shift * 0.5);
    
    // Boost saturation
    float luminance = dot(warm_shift, vec3(0.299, 0.587, 0.114));
    return mix(vec3(luminance), warm_shift, sat_boost);
}

/**
 * SUBTLE MOTION ATTENTION - Creates gentle drift that keeps eyes engaged
 */
vec2 effect_attention_drift(vec2 input_uv, float time) {
    // Very subtle parallax-like movement
    float drift_x = sin(time * 0.2) * 0.01;  // Super subtle horizontal drift
    float drift_y = cos(time * 0.15) * 0.005; // Even more subtle vertical
    
    // Add slight rotation (barely noticeable but subconsciously engaging)
    float angle = sin(time * 0.1) * 0.002; // 0.002 radians = ~0.1 degrees
    
    vec2 centered_uv = input_uv - 0.5;
    vec2 rotated_uv = vec2(
        centered_uv.x * cos(angle) - centered_uv.y * sin(angle),
        centered_uv.x * sin(angle) + centered_uv.y * cos(angle)
    );
    
    return rotated_uv + 0.5 + vec2(drift_x, drift_y);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
void main() {
    // Start with original UV coordinates
    vec2 working_uv = uv;
    
    // Apply retention-boosting effects
    working_uv = effect_ken_burns(working_uv, u_time);        // Golden ratio movement
    working_uv = effect_hypnotic_pulse(working_uv, u_time);   // Subtle pulse
    working_uv = effect_attention_drift(working_uv, u_time);  // Micro-movements
    working_uv = effect_crt_distortion(working_uv);           // CRT effect
    
    // Bounds check
    if (working_uv.x < -0.1 || working_uv.x > 1.1 || working_uv.y < -0.1 || working_uv.y > 1.1) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Sample texture
    vec2 clamped_uv = clamp(working_uv, 0.0, 1.0);
    vec4 base_texture = texture(tex, clamped_uv);
    vec3 working_color = base_texture.rgb;
    
    // Apply visual retention effects
    working_color = effect_retention_boost(working_color, u_time); // Color psychology
    working_color = effect_scanlines(working_color, uv, u_time);   // CRT nostalgia
    working_color = effect_vignette(working_color, uv);            // Focus attention
    
    // Subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, base_texture.a);
}