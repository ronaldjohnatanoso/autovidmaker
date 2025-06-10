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
// SAFE PSYCHEDELIC EFFECTS - TRIPPY BUT NOT EPILEPTIC
// ============================================================================

/**
 * GENTLE FRACTAL BREATHING - Slow, hypnotic breathing
 */
vec2 effect_gentle_breathing(vec2 input_uv, float time) {
    vec2 center = vec2(0.5);
    vec2 centered_uv = input_uv - center;
    
    // MUCH slower breathing frequencies (safe for epilepsy)
    float breath1 = sin(time * 0.3) * 0.05;  // Very slow primary breath
    float breath2 = sin(time * 0.7) * 0.02;  // Gentle secondary
    float breath3 = sin(time * 0.1) * 0.08;  // Super slow deep breath
    
    float radius = length(centered_uv);
    float breathing_factor = 1.0 + breath1 + breath2 * sin(radius * 5.0) + breath3 * cos(radius * 3.0);
    
    return center + centered_uv * breathing_factor;
}

/**
 * FLOWING GEOMETRIC PATTERNS - Smooth, flowing sacred geometry
 */
vec3 effect_flowing_geometry(vec3 input_color, vec2 uv_coords, float time) {
    vec2 center = vec2(0.5);
    vec2 pos = uv_coords - center;
    
    float radius = length(pos);
    float angle = atan(pos.y, pos.x);
    
    // Smooth, slow geometric patterns (no strobing)
    float pattern1 = sin(radius * 8.0 + time * 0.5) * sin(angle * 4.0);
    float pattern2 = cos(radius * 12.0 - time * 0.3) * cos(angle * 6.0);
    float mandala = sin(angle * 6.0 + time * 0.2) * cos(radius * 15.0 + time * 0.4);
    
    // Gentle intensity
    float geometric_intensity = (pattern1 + pattern2 + mandala) * 0.08;
    
    // Smooth color shifts (no rapid flashing)
    vec3 rainbow = vec3(
        sin(time * 0.8 + radius * 5.0) * 0.3 + 0.7,
        sin(time * 0.8 + radius * 5.0 + 2.094) * 0.3 + 0.7,
        sin(time * 0.8 + radius * 5.0 + 4.188) * 0.3 + 0.7
    );
    
    return input_color + rainbow * geometric_intensity;
}

/**
 * SMOOTH KALEIDOSCOPE - Gentle kaleidoscope without rapid changes
 */
vec2 effect_smooth_kaleidoscope(vec2 input_uv, float time) {
    vec2 center = vec2(0.5);
    vec2 pos = input_uv - center;
    
    float radius = length(pos);
    float angle = atan(pos.y, pos.x);
    
    // Slow-changing kaleidoscope (4-6 segments, slow transition)
    float segments = 5.0 + sin(time * 0.1) * 1.0; // Very slow segment change
    angle = mod(angle, 6.28318 / segments);
    
    if (mod(floor(angle * segments / 6.28318), 2.0) == 1.0) {
        angle = 6.28318 / segments - angle;
    }
    
    // Very slow rotation
    angle += time * 0.1;
    
    return center + vec2(cos(angle), sin(angle)) * radius;
}

/**
 * LIQUID FLOW - Gentle flowing distortion like liquid
 */
vec2 effect_liquid_flow(vec2 input_uv, float time) {
    // Gentle flowing distortion (no rapid changes)
    float flow_strength = 0.03;
    float flow_frequency = 2.0;
    float flow_speed = 0.5;  // Much slower
    
    float flow_x = sin(input_uv.y * flow_frequency + time * flow_speed) * flow_strength;
    float flow_y = cos(input_uv.x * flow_frequency + time * flow_speed * 0.7) * flow_strength;
    
    return vec2(input_uv.x + flow_x, input_uv.y + flow_y);
}

/**
 * SMOOTH COLOR CYCLING - Gentle color temperature shifts
 */
vec3 effect_smooth_colors(vec3 input_color, float time) {
    // Gentle hue shifting (no rapid cycling)
    float hue_shift = sin(time * 0.4) * 0.5; // Much slower
    
    // Gentle saturation boost
    float saturation_boost = 1.2 + sin(time * 0.3) * 0.2; // 1.0 to 1.4
    
    // Smooth color temperature shift
    vec3 shifted_color = input_color;
    shifted_color.r = input_color.r * (1.0 + hue_shift * 0.3);
    shifted_color.g = input_color.g;
    shifted_color.b = input_color.b * (1.0 - hue_shift * 0.2);
    
    // Gentle saturation
    float luminance = dot(shifted_color, vec3(0.299, 0.587, 0.114));
    return mix(vec3(luminance), shifted_color, saturation_boost);
}

/**
 * GENTLE VISUAL TEXTURE - Subtle texture overlay (no static)
 */
vec3 effect_gentle_texture(vec3 input_color, vec2 uv_coords, float time) {
    // Very subtle texture noise (no rapid flickering)
    float texture1 = sin(uv_coords.x * 20.0 + time * 2.0) * cos(uv_coords.y * 20.0 + time * 1.5);
    float texture2 = sin(uv_coords.x * 40.0 + time * 1.0) * cos(uv_coords.y * 30.0 + time * 0.8);
    
    float gentle_texture = (texture1 + texture2) * 0.02; // Very subtle
    
    return input_color + vec3(gentle_texture);
}

/**
 * SOFT EDGE GLOW - Gentle edge enhancement
 */
vec3 effect_soft_glow(sampler2D tex, vec2 uv_coords, float time) {
    float edge_strength = 0.001;
    
    vec3 center = texture(tex, uv_coords).rgb;
    vec3 up = texture(tex, uv_coords + vec2(0.0, edge_strength)).rgb;
    vec3 down = texture(tex, uv_coords - vec2(0.0, edge_strength)).rgb;
    vec3 left = texture(tex, uv_coords - vec2(edge_strength, 0.0)).rgb;
    vec3 right = texture(tex, uv_coords + vec2(edge_strength, 0.0)).rgb;
    
    vec3 edge = abs(center - up) + abs(center - down) + abs(center - left) + abs(center - right);
    
    // Gentle, slow-changing glow
    vec3 glow_color = vec3(
        sin(time * 1.0) * 0.2 + 0.8,
        sin(time * 1.0 + 2.094) * 0.2 + 0.8,
        sin(time * 1.0 + 4.188) * 0.2 + 0.8
    );
    
    return center + edge * glow_color * 0.5; // Much gentler
}

/**
 * SMOOTH COLOR TRAILS - Gentle trailing effect
 */
vec3 effect_smooth_trails(sampler2D tex, vec2 uv_coords, float time) {
    vec3 color_accumulate = vec3(0.0);
    
    // Fewer, more subtle trails
    for (int i = 0; i < 3; i++) {
        float offset_time = float(i) * 0.05;
        
        vec2 trail_uv = uv_coords;
        trail_uv.x += sin(time * 0.5 - offset_time) * 0.005; // Much more subtle
        trail_uv.y += cos(time * 0.3 - offset_time) * 0.003;
        
        vec3 trail_color = texture(tex, trail_uv).rgb;
        
        float trail_intensity = 1.0 - float(i) * 0.1;
        color_accumulate += trail_color * trail_intensity;
    }
    
    return color_accumulate / 3.0;
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
        
        // Gentle subtitle color shift
        vec3 gentle_subtitle = subtitle_sample.rgb;
        gentle_subtitle = effect_smooth_colors(gentle_subtitle, u_time);
        
        return mix(base_color, gentle_subtitle, subtitle_sample.a);
    }
    
    return base_color;
}

void main() {
    vec2 working_uv = uv;
    
    // Apply GENTLE PSYCHEDELIC transformations
    working_uv = effect_gentle_breathing(working_uv, u_time);
    working_uv = effect_smooth_kaleidoscope(working_uv, u_time);
    working_uv = effect_liquid_flow(working_uv, u_time);
    
    // Bounds check
    if (working_uv.x < -0.3 || working_uv.x > 1.3 || working_uv.y < -0.3 || working_uv.y > 1.3) {
        // Gentle void color (no rapid flashing)
        vec3 void_color = vec3(
            sin(u_time * 0.5 + uv.x * 3.0) * 0.1 + 0.05,
            sin(u_time * 0.5 + uv.y * 3.0 + 2.094) * 0.1 + 0.05,
            sin(u_time * 0.5 + (uv.x + uv.y) * 2.0 + 4.188) * 0.1 + 0.05
        );
        color = vec4(void_color, 1.0);
        return;
    }
    
    vec2 clamped_uv = clamp(working_uv, 0.0, 1.0);
    
    // Sample with gentle trails
    vec3 working_color = effect_smooth_trails(tex, clamped_uv, u_time);
    
    // Apply SAFE PSYCHEDELIC effects
    working_color = effect_soft_glow(tex, clamped_uv, u_time);
    working_color = effect_flowing_geometry(working_color, uv, u_time);
    working_color = effect_smooth_colors(working_color, u_time);
    working_color = effect_gentle_texture(working_color, uv, u_time);
    
    // Subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}