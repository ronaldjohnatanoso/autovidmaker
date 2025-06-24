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
// NOISE FUNCTIONS
// ============================================================================

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// ============================================================================
// FIRE SPARKS EFFECT
// ============================================================================

vec3 effect_fire_sparks(vec3 input_color, vec2 uv_coords, float time) {
    vec3 result = input_color;
    
    // Create multiple layers of sparks
    for (int i = 0; i < 25; i++) {
        float seed = float(i) * 123.456;
        
        // Random spark initial position (start from bottom)
        vec2 spark_pos = vec2(
            0.3 + random(vec2(seed, 0.0)) * 0.4, // Center horizontally
            -0.1 + random(vec2(0.0, seed)) * 0.2  // Start from bottom
        );
        
        // Spark movement - rising upward with wind drift
        float speed = 0.4 + random(vec2(seed, seed)) * 0.6;
        float life_time = mod(time + seed * 10.0, 8.0); // Each spark has 8 second cycle
        
        spark_pos.y += life_time * speed;
        spark_pos.x += sin(life_time * 1.5 + seed) * 0.1; // Wind drift
        
        // Skip if spark is out of bounds
        if (spark_pos.y > 1.2) continue;
        
        // Distance from current pixel to spark
        float dist = distance(uv_coords, spark_pos);
        
        // Much larger and more visible sparks
        float spark_size = 0.008 + random(vec2(seed * 2.0, seed)) * 0.006;
        float intensity = 1.0 - smoothstep(0.0, spark_size, dist);
        
        // Spark fades as it rises
        float fade = 1.0 - smoothstep(0.0, 1.0, life_time / 6.0);
        intensity *= fade;
        
        // Strong flickering effect
        float flicker = 0.6 + 0.4 * sin(time * 15.0 + seed * 15.0);
        intensity *= flicker;
        
        // Bright fire colors - yellow/orange/red
        vec3 spark_color;
        float color_rand = random(vec2(seed * 3.0, seed));
        if (color_rand < 0.3) {
            spark_color = vec3(1.0, 1.0, 0.3);  // Bright yellow
        } else if (color_rand < 0.7) {
            spark_color = vec3(1.0, 0.6, 0.1);  // Orange
        } else {
            spark_color = vec3(1.0, 0.2, 0.0);  // Red
        }
        
        // Add bright spark to result
        result += spark_color * intensity * 2.5; // Much brighter
    }
    
    return result;
}

// ============================================================================
// SMOKE OVERLAY EFFECT
// ============================================================================

vec3 effect_smoke_overlay(vec3 input_color, vec2 uv_coords, float time) {
    // Create multiple layers of smoke rising from bottom
    vec2 smoke_uv = uv_coords;
    
    // Make smoke rise from bottom
    smoke_uv.y = (smoke_uv.y - time * 0.1) * 2.0; // Rising motion
    smoke_uv.x *= 4.0;
    
    // Multiple noise layers for realistic smoke
    float smoke1 = noise(smoke_uv + vec2(time * 0.3, 0.0));
    float smoke2 = noise(smoke_uv * 1.5 + vec2(-time * 0.2, time * 0.4));
    float smoke3 = noise(smoke_uv * 0.8 + vec2(time * 0.15, -time * 0.25));
    
    // Combine smoke layers
    float smoke_density = (smoke1 + smoke2 * 0.7 + smoke3 * 0.5) / 2.2;
    
    // Strong vertical gradient - thick at bottom, thin at top
    float vertical_mask = 1.0 - smoothstep(0.0, 0.7, uv_coords.y);
    smoke_density *= vertical_mask;
    
    // Make smoke more visible
    smoke_density = smoothstep(0.2, 0.9, smoke_density);
    
    // Dark smoke color
    vec3 smoke_color = vec3(0.02, 0.02, 0.02); // Very dark
    
    // Apply heavy smoke overlay
    return mix(input_color, smoke_color, smoke_density * 0.8); // Much stronger
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
    
    // Apply smoke overlay first (behind sparks)
    working_color = effect_smoke_overlay(working_color, uv, u_time);
    
    // Apply fire sparks effect
    working_color = effect_fire_sparks(working_color, uv, u_time);
    
    // Add subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}