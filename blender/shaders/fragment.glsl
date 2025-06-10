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
// SIMPLE RED PULSE EFFECT
// ============================================================================

/**
 * RED PULSE - Simple red overlay that pulses
 */
vec3 effect_red_pulse(vec3 input_color, float time) {
    // Create a pulsing red overlay
    float pulse = sin(time * 2.0) * 0.5 + 0.5; // 0.0 to 1.0
    
    // Red tint that varies with pulse
    vec3 red_tint = vec3(1.0, 0.3, 0.3) * pulse * 0.3; // Adjust intensity here
    
    return input_color + red_tint;
}

// ============================================================================
// SUBTITLE OVERLAY
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

// ============================================================================
// MAIN FUNCTION
// ============================================================================

void main() {
    // Sample the original texture
    vec3 working_color = texture(tex, uv).rgb;
    
    // Apply red pulse effect
    working_color = effect_red_pulse(working_color, u_time);
    
    // Add subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}