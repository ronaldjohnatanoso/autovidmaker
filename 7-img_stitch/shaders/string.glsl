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
// SIMPLE BUT COOL STRING EFFECT
// ============================================================================

vec3 effect_string_waves(vec2 uv_coords, float time) {
    // Center coordinates
    vec2 p = (uv_coords - 0.5) * 2.0;
    
    // Create string-like waves
    float wave1 = sin(p.x * 10.0 + time * 3.0) * 0.1;
    float wave2 = sin(p.x * 15.0 - time * 4.0) * 0.05;
    float wave3 = sin(p.x * 20.0 + time * 2.0) * 0.02;
    
    // Combine waves
    float y_offset = wave1 + wave2 + wave3;
    
    // Distance from wave line
    float dist = abs(p.y - y_offset);
    
    // Create glow effect
    float glow1 = 1.0 / (1.0 + dist * 50.0);
    float glow2 = 1.0 / (1.0 + dist * 20.0);
    float glow3 = 1.0 / (1.0 + dist * 10.0);
    
    // Colors
    vec3 color1 = vec3(1.0, 0.5, 0.0) * glow1; // Orange
    vec3 color2 = vec3(0.0, 0.5, 1.0) * glow2; // Blue
    vec3 color3 = vec3(1.0, 0.0, 0.5) * glow3; // Pink
    
    return color1 + color2 * 0.5 + color3 * 0.3;
}

vec3 effect_multiple_strings(vec2 uv_coords, float time) {
    vec3 final_color = vec3(0.0);
    
    // Create multiple string layers
    for (int i = 0; i < 5; i++) {
        vec2 p = (uv_coords - 0.5) * 2.0;
        
        // Offset each string
        p.y += float(i) * 0.3 - 0.6;
        
        // Different frequencies for each string
        float freq = 8.0 + float(i) * 3.0;
        float speed = 2.0 + float(i) * 0.5;
        
        // Wave equation
        float wave = sin(p.x * freq + time * speed) * 0.08;
        
        // Distance to string
        float dist = abs(p.y - wave);
        
        // Glow
        float glow = 1.0 / (1.0 + dist * 40.0);
        
        // Color based on string index
        vec3 string_color = vec3(
            sin(float(i) + time) * 0.5 + 0.5,
            cos(float(i) + time * 1.2) * 0.5 + 0.5,
            sin(float(i) * 2.0 + time * 0.8) * 0.5 + 0.5
        );
        
        final_color += string_color * glow * 0.3;
    }
    
    return final_color;
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
    vec3 original_color = texture(tex, uv).rgb;
    
    // Create string effect - using the multiple strings version
    vec3 string_effect = effect_multiple_strings(uv, u_time);
    
    // Blend original with string effect at 70% opacity (more visible)
    vec3 final_color = mix(original_color, string_effect, 0.7);
    
    // Add subtitles
    final_color = add_subtitle_overlay(final_color, uv);
    
    color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
}