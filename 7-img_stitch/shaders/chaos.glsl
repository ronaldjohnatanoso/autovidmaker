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
// SAND DISSOLUTION AND SPEED STREAMING EFFECTS
// ============================================================================

/**
 * NOISE FUNCTION - Pseudo-random noise for particle effects
 */
float noise(vec2 pos) {
    return fract(sin(dot(pos, vec2(12.9898, 78.233))) * 43758.5453);
}

/**
 * SAND PARTICLE DISSOLUTION - Makes pixels break apart into sand-like particles
 */
vec4 effect_sand_dissolution(sampler2D tex, vec2 uv_coords, float time) {
    vec2 original_uv = uv_coords;
    
    // Create noise-based particle positions
    float particle_noise = noise(uv_coords * 100.0 + time * 0.5);
    float dissolve_threshold = 0.1 + sin(time * 2.0) * 0.05; // Much weaker dissolution
    
    // Different dissolution patterns based on position
    float dissolve_pattern = 
        noise(uv_coords * 50.0) * 0.3 +
        noise(uv_coords * 25.0 + time * 0.8) * 0.3 +
        noise(uv_coords * 12.5 + time * 1.2) * 0.4;
    
    // Areas that should dissolve completely (much less)
    if (dissolve_pattern < dissolve_threshold) {
        return vec4(0.0, 0.0, 0.0, 0.0); // Transparent (dissolved)
    }
    
    // Particle streaming effect - make remaining pixels flow (reduced)
    vec2 flow_direction = vec2(1.0, 0.0); // Main flow to the right
    float flow_speed = time * 3.0;
    
    // Add turbulence to the flow (reduced)
    flow_direction.y += sin(uv_coords.x * 20.0 + flow_speed) * 0.1;
    flow_direction.x += cos(uv_coords.y * 15.0 + flow_speed * 0.8) * 0.05;
    
    // Offset UV based on flow and particle noise (much smaller)
    vec2 flow_uv = uv_coords + flow_direction * particle_noise * 0.02;
    
    // Sample the texture
    vec4 tex_color = texture(tex, clamp(flow_uv, 0.0, 1.0));
    
    // Make particles fade based on dissolution strength
    float particle_alpha = smoothstep(dissolve_threshold, dissolve_threshold + 0.2, dissolve_pattern);
    tex_color.a *= particle_alpha;
    
    return tex_color;
}

/**
 * SPEED STREAMING - Creates motion blur and streaming trails
 */
vec3 effect_speed_streaming(sampler2D tex, vec2 uv_coords, float time) {
    vec3 color_accumulate = vec3(0.0);
    int samples = 4; // Reduced samples
    
    // Stream direction (mainly horizontal, like wind)
    vec2 stream_dir = vec2(1.0, 0.0);
    
    // Add some vertical variation to streams (reduced)
    stream_dir.y += sin(uv_coords.x * 10.0 + time * 4.0) * 0.1;
    stream_dir.y += cos(uv_coords.y * 8.0 + time * 3.0) * 0.05;
    
    // Sample multiple positions along the stream
    for (int i = 0; i < samples; i++) {
        float offset_factor = float(i) / float(samples - 1);
        
        // Stream distance varies by noise (much smaller)
        float stream_length = 0.05 + noise(uv_coords * 30.0 + time) * 0.03;
        vec2 stream_offset = stream_dir * stream_length * offset_factor;
        
        vec2 sample_uv = uv_coords - stream_offset;
        
        // Add some jitter to make it more particle-like (reduced)
        sample_uv += (noise(sample_uv * 50.0 + time * 2.0) - 0.5) * 0.005;
        
        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            vec4 stream_sample = effect_sand_dissolution(tex, sample_uv, time);
            
            // Weight samples - more recent (closer) samples are stronger
            float weight = 1.0 - offset_factor * 0.3;
            color_accumulate += stream_sample.rgb * stream_sample.a * weight;
        }
    }
    
    return color_accumulate / float(samples);
}

/**
 * PARTICLE TRAILS - Creates individual particle trails streaming away
 */
vec3 effect_particle_trails(sampler2D tex, vec2 uv_coords, float time) {
    vec3 final_color = vec3(0.0);
    
    // Create multiple trail layers (reduced)
    for (int layer = 0; layer < 2; layer++) {
        float layer_offset = float(layer) * 0.1;
        
        // Different trail directions for each layer (reduced movement)
        vec2 trail_dir = vec2(
            1.0 + sin(time * 2.0 + layer_offset) * 0.2,
            sin(time * 3.0 + layer_offset + uv_coords.x * 5.0) * 0.3
        );
        
        // Trail length varies (shorter)
        float trail_length = 0.08 + sin(time * 1.5 + layer_offset) * 0.03;
        
        // Sample along the trail (fewer samples)
        for (int i = 0; i < 3; i++) {
            float trail_pos = float(i) / 2.0;
            vec2 trail_uv = uv_coords - trail_dir * trail_length * trail_pos;
            
            // Add particle-like jitter (reduced)
            trail_uv += (noise(trail_uv * 40.0 + time * 3.0 + layer_offset) - 0.5) * 0.01;
            
            if (trail_uv.x >= 0.0 && trail_uv.x <= 1.0 && trail_uv.y >= 0.0 && trail_uv.y <= 1.0) {
                vec4 trail_sample = effect_sand_dissolution(tex, trail_uv, time + layer_offset);
                
                // Fade trail based on distance (less intense)
                float trail_fade = (1.0 - trail_pos) * 0.1;
                final_color += trail_sample.rgb * trail_sample.a * trail_fade;
            }
        }
    }
    
    return final_color;
}

/**
 * WIND DISTORTION - Adds wind-like distortion to the streaming effect
 */
vec2 effect_wind_distortion(vec2 uv_coords, float time) {
    vec2 wind_uv = uv_coords;
    
    // Primary wind direction (horizontal) - much reduced
    wind_uv.x += sin(uv_coords.y * 15.0 + time * 5.0) * 0.01;
    wind_uv.y += cos(uv_coords.x * 12.0 + time * 4.0) * 0.008;
    
    // Secondary turbulence - reduced
    wind_uv.x += sin(uv_coords.y * 30.0 + time * 8.0) * 0.005;
    wind_uv.y += cos((uv_coords.x + uv_coords.y) * 20.0 + time * 6.0) * 0.005;
    
    // Fine detail turbulence - reduced
    wind_uv += (noise(uv_coords * 100.0 + time * 2.0) - 0.5) * 0.003;
    
    return wind_uv;
}

/**
 * CLEAN SUBTITLE RENDERING - No effects on subtitles
 */
vec3 add_clean_subtitle(vec3 base_color, vec2 uv_coords) {
    if (show_subtitle < 0.5 || subtitle_size.x <= 0.0 || subtitle_size.y <= 0.0) {
        return base_color;
    }
    
    // Use original UV coordinates (no distortion for subtitles)
    vec2 subtitle_uv = (uv_coords - subtitle_pos) / subtitle_size;
    
    if (subtitle_uv.x >= 0.0 && subtitle_uv.x <= 1.0 && subtitle_uv.y >= 0.0 && subtitle_uv.y <= 1.0) {
        vec4 subtitle_sample = texture(subtitle_tex, subtitle_uv);
        
        // Simple clean blend - no effects
        return mix(base_color, subtitle_sample.rgb, subtitle_sample.a);
    }
    
    return base_color;
}

// ============================================================================
// MAIN STREAMING SAND EFFECT
// ============================================================================

void main() {
    // Apply wind distortion to UV coordinates
    vec2 distorted_uv = effect_wind_distortion(uv, u_time);
    
    // Get the original image first
    vec4 original_sample = texture(tex, distorted_uv);
    
    // Get the main streaming effect
    vec3 streaming_color = effect_speed_streaming(tex, distorted_uv, u_time);
    
    // Add particle trails
    vec3 trail_color = effect_particle_trails(tex, distorted_uv, u_time);
    
    // Combine streaming and trails (much weaker)
    vec3 effect_color = streaming_color + trail_color * 0.2;
    
    // Start with original image and blend in effects
    vec3 final_color = original_sample.rgb;
    
    // Add dissolution effect to original (very light)
    vec4 base_sample = effect_sand_dissolution(tex, distorted_uv, u_time);
    final_color = mix(final_color, base_sample.rgb, 0.2);
    
    // Blend in the streaming effects (very light)
    final_color = mix(final_color, effect_color, 0.15);
    
    // Add clean subtitles (no effects applied)
    final_color = add_clean_subtitle(final_color, uv);
    
    // Enhance the streaming effect with some glow (much reduced)
    final_color *= 1.0 + sin(u_time * 4.0) * 0.03;
    
    color = vec4(final_color, 1.0);
}