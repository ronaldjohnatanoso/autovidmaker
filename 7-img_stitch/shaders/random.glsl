#version 330
uniform sampler2D tex;           
uniform sampler2D subtitle_tex;  
uniform float u_time;            
uniform vec2 subtitle_size;      
uniform vec2 subtitle_pos;       
uniform float show_subtitle;     
in vec2 uv;                      
out vec4 color;                  

// Improved hash function for better randomness
vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123);
}

// Conway's Game of Life implementation
float gameOfLife(vec2 coord, float time) {
    // Grid resolution
    float grid_size = 100.0;
    vec2 grid_coord = floor(coord * grid_size);
    
    // Time-based evolution (update every 0.1 seconds)
    float time_step = floor(time * 10.0);
    
    // Generate initial state based on grid position and time
    vec3 seed = vec3(grid_coord, time_step);
    float current_state = step(0.6, hash3(seed).x);
    
    // Count neighbors
    float neighbors = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) continue;
            
            vec2 neighbor_coord = grid_coord + vec2(float(x), float(y));
            vec3 neighbor_seed = vec3(neighbor_coord, time_step);
            neighbors += step(0.6, hash3(neighbor_seed).x);
        }
    }
    
    // Apply Conway's rules
    float next_state = 0.0;
    
    if (current_state > 0.5) {
        // Cell is alive
        if (neighbors == 2.0 || neighbors == 3.0) {
            next_state = 1.0; // Survives
        }
    } else {
        // Cell is dead
        if (neighbors == 3.0) {
            next_state = 1.0; // Born
        }
    }
    
    // Smooth interpolation for visual appeal
    vec2 fract_coord = fract(coord * grid_size);
    float cell_smoothness = smoothstep(0.1, 0.9, fract_coord.x) * smoothstep(0.1, 0.9, fract_coord.y);
    
    return next_state * cell_smoothness;
}

// Generate Game of Life overlay
vec3 apply_game_of_life(vec3 base_color, vec2 uv, float time) {
    // Create multiple layers with different scales and colors
    float life1 = gameOfLife(uv + vec2(0.0, 0.0), time);
    float life2 = gameOfLife(uv * 1.5 + vec2(100.0, 50.0), time * 1.2);
    float life3 = gameOfLife(uv * 0.7 + vec2(200.0, 300.0), time * 0.8);
    
    // Combine layers with different colors
    vec3 life_color = vec3(0.0);
    life_color += life1 * vec3(0.0, 1.0, 0.3);  // Green
    life_color += life2 * vec3(0.3, 0.0, 1.0);  // Blue
    life_color += life3 * vec3(1.0, 0.2, 0.0);  // Red
    
    // Add some dynamic color shifting
    float color_shift = sin(time * 2.0) * 0.1 + 0.9;
    life_color *= color_shift;
    
    // Mix with base color at 50% opacity
    return mix(base_color, base_color + life_color, 0.5);
}

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

void main() {
    // Sample the base video texture
    vec3 base_color = texture(tex, uv).rgb;
    
    // Apply Conway's Game of Life overlay
    vec3 enhanced_color = apply_game_of_life(base_color, uv, u_time);
    
    // Add subtitles
    vec3 final_color = add_subtitle_overlay(enhanced_color, uv);
    
    color = vec4(final_color, 1.0);
}