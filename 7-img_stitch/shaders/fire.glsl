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
    for (int i = 0; i < 15; i++) {
        float seed = float(i) * 123.456;
        
        // Random spark position
        vec2 spark_pos = vec2(
            random(vec2(seed, 0.0)),
            random(vec2(0.0, seed))
        );
        
        // Spark movement - floating upward with some drift
        float speed = 0.3 + random(vec2(seed, seed)) * 0.5;
        spark_pos.y = mod(spark_pos.y + time * speed, 1.2) - 0.1;
        spark_pos.x += sin(time * 2.0 + seed) * 0.02; // Small horizontal drift
        
        // Distance from current pixel to spark
        float dist = distance(uv_coords, spark_pos);
        
        // Spark size and intensity
        float spark_size = 0.003 + random(vec2(seed * 2.0, seed)) * 0.002;
        float intensity = 1.0 - smoothstep(0.0, spark_size, dist);
        
        // Flickering effect
        float flicker = 0.8 + 0.2 * sin(time * 10.0 + seed * 10.0);
        intensity *= flicker;
        
        // Spark color - orange to red gradient
        vec3 spark_color = mix(
            vec3(1.0, 0.3, 0.1),  // Red
            vec3(1.0, 0.7, 0.2),  // Orange
            random(vec2(seed * 3.0, seed))
        );
        
        // Add spark to result
        result += spark_color * intensity * 0.8;
    }
    
    return result;
}

// ============================================================================
// SMOKE OVERLAY EFFECT
// ============================================================================

vec3 effect_smoke_overlay(vec3 input_color, vec2 uv_coords, float time) {
    // Create multiple layers of smoke
    vec2 smoke_uv = uv_coords * 3.0;
    
    // Animated smoke patterns
    float smoke1 = noise(smoke_uv + vec2(time * 0.1, time * 0.2));
    float smoke2 = noise(smoke_uv * 2.0 + vec2(-time * 0.15, time * 0.1));
    float smoke3 = noise(smoke_uv * 0.5 + vec2(time * 0.05, -time * 0.3));
    
    // Combine smoke layers
    float smoke_density = (smoke1 + smoke2 * 0.5 + smoke3 * 0.3) / 1.8;
    
    // Make smoke more concentrated at bottom and dispersed at top
    float vertical_gradient = 1.0 - smoothstep(0.2, 0.9, uv_coords.y);
    smoke_density *= vertical_gradient;
    
    // Smooth the smoke
    smoke_density = smoothstep(0.3, 0.8, smoke_density);
    
    // Smoke color - dark gray to black
    vec3 smoke_color = vec3(0.1, 0.1, 0.1);
    
    // Apply smoke overlay
    return mix(input_color, smoke_color, smoke_density * 0.4);
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