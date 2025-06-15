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
// ANALOG HORROR EFFECTS
// ============================================================================

// Random function for noise
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

// VHS-style noise
float vhs_noise(vec2 uv, float time) {
    float noise = random(uv + time * 0.1);
    return noise * 0.1;
}

// Static interference
float static_interference(vec2 uv, float time) {
    float static_noise = random(uv * 100.0 + time * 50.0);
    float interference = step(0.95, static_noise) * 0.8;
    return interference;
}

// TV frame distortion
vec2 tv_distortion(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    vec2 offset = uv - center;
    float dist = length(offset);
    
    // Enhanced barrel distortion for more pronounced curved TV effect
    float distortion = 1.0 + 0.3 * dist * dist + 0.1 * dist * dist * dist * dist;
    
    // Add slight pincushion effect at the edges
    vec2 distorted = center + offset * distortion;
    
    // Add corner rounding effect
    vec2 corner_effect = abs(offset) * abs(offset);
    distorted += corner_effect * 0.05;
    
    return distorted;
}

// Safe texture sampling that returns black for out-of-bounds coordinates
vec3 sample_texture_safe(sampler2D tex, vec2 uv) {
    // Check if coordinates are within valid range
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec3(0.0, 0.0, 0.0); // Return black for out-of-bounds
    }
    return texture(tex, uv).rgb;
}

// Updated chromatic aberration with safe sampling
vec3 chromatic_aberration(sampler2D tex, vec2 uv, float intensity) {
    vec2 offset = vec2(intensity * 0.01, 0.0);
    float r = sample_texture_safe(tex, uv + offset).r;
    float g = sample_texture_safe(tex, uv).g;
    float b = sample_texture_safe(tex, uv - offset).b;
    return vec3(r, g, b);
}

// Scanlines effect
float scanlines(vec2 uv, float time) {
    float line = sin(uv.y * 800.0 + time * 2.0) * 0.04;
    return 1.0 - abs(line);
}

// Flickering effect
float flicker(float time) {
    float flicker_speed = 20.0;
    float flicker_intensity = 0.3;
    return 1.0 - (random(vec2(time)) * flicker_intensity * step(0.9, sin(time * flicker_speed)));
}

// Desaturation with slight color shift
vec3 horror_color_grade(vec3 color) {
    // Desaturate
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 desaturated = mix(color, vec3(gray), 0.4);
    
    // Add slight green/blue tint for that eerie look
    desaturated.g *= 1.1;
    desaturated.b *= 0.9;
    desaturated.r *= 0.95;
    
    return desaturated;
}

// Glitch displacement
vec2 glitch_displacement(vec2 uv, float time) {
    float glitch_line = step(0.98, random(vec2(floor(uv.y * 50.0), time * 0.1)));
    vec2 displacement = vec2(glitch_line * (random(vec2(time)) - 0.5) * 0.1, 0.0);
    return uv + displacement;
}

// Vignette effect
float vignette(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    return 1.0 - smoothstep(0.3, 0.8, dist);
}

// ============================================================================
// SUBTITLE OVERLAY
// ============================================================================

vec3 add_subtitle_overlay(vec3 base_color, vec2 uv_coords, float time) {
    if (show_subtitle < 0.5 || subtitle_size.x <= 0.0 || subtitle_size.y <= 0.0) {
        return base_color;
    }
    
    vec2 subtitle_uv = (uv_coords - subtitle_pos) / subtitle_size;
    
    if (subtitle_uv.x >= 0.0 && subtitle_uv.x <= 1.0 && subtitle_uv.y >= 0.0 && subtitle_uv.y <= 1.0) {
        // Apply same distortions to subtitle UV coordinates
        vec2 distorted_subtitle_uv = tv_distortion(subtitle_uv);
        distorted_subtitle_uv = glitch_displacement(distorted_subtitle_uv, time);
        
        // Sample subtitle with chromatic aberration
        vec3 subtitle_color = chromatic_aberration(subtitle_tex, distorted_subtitle_uv, sin(time * 2.0) * 0.3 + 0.3).rgb;
        
        // Get alpha from original coordinates for proper masking
        float subtitle_alpha = texture(subtitle_tex, subtitle_uv).a;
        
        // Apply horror effects to subtitle
        subtitle_color += vec3(vhs_noise(subtitle_uv, time) * 0.5);
        subtitle_color = mix(subtitle_color, vec3(1.0), static_interference(subtitle_uv, time) * 0.3);
        subtitle_color *= scanlines(subtitle_uv, time);
        subtitle_color *= flicker(time);
        
        // Horror color grading for subtitle
        float gray = dot(subtitle_color, vec3(0.299, 0.587, 0.114));
        subtitle_color = mix(subtitle_color, vec3(gray), 0.2);
        subtitle_color.r *= 1.2; // Slightly red-shifted for horror
        subtitle_color.g *= 0.8;
        subtitle_color.b *= 0.7;
        
        // Add eerie glow effect
        float glow_intensity = sin(time * 8.0) * 0.1 + 0.9;
        subtitle_color *= glow_intensity;
        
        return mix(base_color, subtitle_color, subtitle_alpha);
    }
    
    return base_color;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

void main() {
    vec2 working_uv = uv;
    
    // Apply TV distortion
    working_uv = tv_distortion(working_uv);
    
    // Apply glitch displacement
    working_uv = glitch_displacement(working_uv, u_time);
    
    // Sample texture with chromatic aberration using safe sampling
    vec3 working_color = chromatic_aberration(tex, working_uv, sin(u_time * 3.0) * 0.5 + 0.5);
    
    // Add VHS noise
    working_color += vec3(vhs_noise(uv, u_time));
    
    // Add static interference
    working_color = mix(working_color, vec3(1.0), static_interference(uv, u_time));
    
    // Apply scanlines
    working_color *= scanlines(uv, u_time);
    
    // Apply flickering
    working_color *= flicker(u_time);
    
    // Apply horror color grading
    working_color = horror_color_grade(working_color);
    
    // Apply vignette
    working_color *= vignette(uv);
    
    // Add subtitles with horror effects
    working_color = add_subtitle_overlay(working_color, uv, u_time);
    
    // Final output with slight contrast boost
    working_color = pow(working_color, vec3(1.2));
    
    color = vec4(working_color, 1.0);
}