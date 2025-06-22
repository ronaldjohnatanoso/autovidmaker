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
// HOLY HEAVENLY EFFECTS
// ============================================================================

/**
 * DIVINE LIGHT RAYS - Soft radiating beams from center
 */
vec3 effect_divine_rays(vec3 input_color, vec2 uv_coords, float time) {
    vec2 center = vec2(0.5);
    vec2 direction = uv_coords - center;
    float angle = atan(direction.y, direction.x);
    float distance = length(direction);
    
    // Create rotating light rays
    float ray_count = 12.0;
    float ray_angle = angle * ray_count + time * 0.5;
    float ray_intensity = sin(ray_angle) * 0.5 + 0.5;
    ray_intensity = pow(ray_intensity, 3.0); // Make rays more focused
    
    // Fade rays with distance
    ray_intensity *= (1.0 - distance) * 0.3;
    
    // Golden divine light
    vec3 divine_light = vec3(1.0, 0.9, 0.7) * ray_intensity;
    
    return input_color + divine_light;
}

/**
 * HEAVENLY GLOW - Soft ethereal lighting
 */
vec3 effect_heavenly_glow(vec3 input_color, vec2 uv_coords, float time) {
    vec2 center = vec2(0.5);
    float distance = length(uv_coords - center);
    
    // Pulsing heavenly glow
    float glow_pulse = sin(time * 1.5) * 0.1 + 0.9;
    float glow_intensity = (1.0 - distance) * 0.4 * glow_pulse;
    
    // Soft golden-white glow
    vec3 heavenly_color = vec3(1.0, 0.95, 0.8) * glow_intensity;
    
    return input_color + heavenly_color;
}

/**
 * CLOUD VIGNETTE - Soft cloudy edges
 */
vec3 effect_cloud_vignette(vec3 input_color, vec2 uv_coords, float time) {
    vec2 center = uv_coords - 0.5;
    float distance = length(center);
    
    // Create cloud-like noise for soft edges
    float cloud_noise = sin(uv_coords.x * 20.0 + time) * sin(uv_coords.y * 15.0 + time * 1.2);
    cloud_noise += sin(uv_coords.x * 35.0 - time * 0.8) * sin(uv_coords.y * 25.0 - time * 0.5);
    cloud_noise = cloud_noise * 0.1 + 0.9; // Subtle cloud variation
    
    // Soft vignette falloff
    float vignette = 1.0 - smoothstep(0.3, 0.8, distance * cloud_noise);
    vignette = max(vignette, 0.2); // Don't go completely black
    
    // Blend with soft heavenly color at edges
    vec3 cloud_color = vec3(0.9, 0.95, 1.0); // Soft blue-white clouds
    vec3 final_color = mix(cloud_color, input_color, vignette);
    
    return final_color;
}

/**
 * GENTLE FLOATING - Subtle peaceful movement
 */
vec2 effect_gentle_float(vec2 input_uv, float time) {
    // Very subtle floating motion
    vec2 float_offset = vec2(
        sin(time * 0.3) * 0.002,
        cos(time * 0.4) * 0.002
    );
    
    return input_uv + float_offset;
}

/**
 * DIVINE BRIGHTNESS - Enhance the holy atmosphere
 */
vec3 effect_divine_brightness(vec3 input_color, float time) {
    // Gentle brightness pulse
    float brightness_pulse = sin(time * 0.8) * 0.05 + 1.1;
    
    // Enhance whites and lights
    vec3 enhanced = input_color * brightness_pulse;
    
    // Add subtle golden tint
    enhanced.r *= 1.02;
    enhanced.g *= 1.01;
    
    return enhanced;
}

/**
 * KEN BURNS EFFECT - Slow zoom and pan motion
 */
vec2 effect_ken_burns(vec2 input_uv, float time) {
    // Ken Burns parameters
    float zoom_speed = 0.1;        // How fast we zoom
    float pan_speed = 0.05;        // How fast we pan
    float max_zoom = 1.3;          // Maximum zoom level
    float min_zoom = 1.0;          // Minimum zoom level
    
    // Create slow zoom in and out cycle
    float zoom_cycle = sin(time * zoom_speed) * 0.5 + 0.5; // 0 to 1
    float zoom_factor = mix(min_zoom, max_zoom, zoom_cycle);
    
    // Create slow panning motion
    vec2 pan_offset = vec2(
        sin(time * pan_speed) * 0.05,        // Horizontal pan
        cos(time * pan_speed * 0.7) * 0.03   // Vertical pan (different speed)
    );
    
    // Apply zoom (scale from center)
    vec2 centered_uv = input_uv - 0.5;
    centered_uv /= zoom_factor;
    centered_uv += 0.5;
    
    // Apply pan
    centered_uv += pan_offset;
    
    return centered_uv;
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
        
        // Golden color for the text
        vec3 golden_color = vec3(1.0, 0.8, 0.3); // Rich golden color
        
        // Use the original luminance to preserve text details and stroke
        float luminance = dot(subtitle_sample.rgb, vec3(0.299, 0.587, 0.114));
        
        // SHINING EFFECT - Enhanced sun-like shine
        // Create a moving shine wave across the text
        float shine_speed = 1.5; // Slightly slower for more dramatic effect
        float shine_width = 0.4; // Wider shine beam
        float shine_position = mod(u_time * shine_speed, 2.5) - 0.75; // Wider sweep range
        
        // Calculate distance from shine position
        float shine_distance = abs(subtitle_uv.x - shine_position);
        float shine_intensity = 1.0 - smoothstep(0.0, shine_width, shine_distance);
        shine_intensity = pow(shine_intensity, 0.5); // Make shine falloff more gradual
        
        // Add multiple layers of glow for sun-like effect
        float glow_intensity = exp(-shine_distance * 6.0) * 0.8; // Stronger primary glow
        float outer_glow = exp(-shine_distance * 3.0) * 0.4; // Wider secondary glow
        float far_glow = exp(-shine_distance * 1.5) * 0.2; // Even wider tertiary glow
        
        // Combine all glow layers
        shine_intensity = max(shine_intensity, glow_intensity);
        shine_intensity = max(shine_intensity, outer_glow);
        shine_intensity = max(shine_intensity, far_glow);
        
        // Make shine more intense when it's directly over text
        shine_intensity *= (1.0 + luminance * 0.5); // Brighter on actual text areas
        
        // Enhance the golden color with much brighter shine
        vec3 shine_color = vec3(1.5, 1.3, 0.9); // Brighter, more golden shine
        vec3 enhanced_golden = mix(golden_color, shine_color, shine_intensity * 0.9); // Stronger mix
        
        // Add extra brightness pulse for divine effect (more intense)
        float divine_pulse = sin(u_time * 3.0) * 0.2 + 1.2; // Stronger pulse
        enhanced_golden *= divine_pulse;
        
        // Add sun-like bloom effect
        float bloom_effect = shine_intensity * 0.3;
        enhanced_golden += vec3(bloom_effect * 1.2, bloom_effect * 0.9, bloom_effect * 0.6);
        
        // Apply the enhanced color
        vec3 golden_text = enhanced_golden * luminance;
        
        // Enhance the alpha blending for stronger shine effect
        float enhanced_alpha = subtitle_sample.a * (1.0 + shine_intensity * 0.6); // Stronger alpha boost
        enhanced_alpha = clamp(enhanced_alpha, 0.0, 1.0);
        
        return mix(base_color, golden_text, enhanced_alpha);
    }
    
    return base_color;
}

void main() {
    vec2 working_uv = uv;
    
    // Apply gentle holy transformations
    working_uv = effect_gentle_float(working_uv, u_time);
    
    // Apply Ken Burns effect
    working_uv = effect_ken_burns(working_uv, u_time);
    
    // Sample the texture
    vec3 working_color = texture(tex, working_uv).rgb;
    
    // Apply heavenly effects
    working_color = effect_divine_brightness(working_color, u_time);
    working_color = effect_heavenly_glow(working_color, uv, u_time);
    working_color = effect_divine_rays(working_color, uv, u_time);
    working_color = effect_cloud_vignette(working_color, uv, u_time);
    
    // Subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}