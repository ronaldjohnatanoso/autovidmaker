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
// HALLUCINATION EFFECTS
// ============================================================================

/**
 * BREATHING WALLS EFFECT
 */
vec2 effect_breathing_walls(vec2 input_uv, float time) {
    vec2 centered = input_uv - 0.5;
    
    // Breathing motion - walls expanding and contracting - reduced intensity
    float breath = sin(time * 1.5) * 0.15 + 0.85; // Reduced from 0.3 + 0.7
    centered *= breath;
    
    // Add wavy distortion like melting walls - reduced intensity
    float wave_x = sin(input_uv.y * 10.0 + time * 2.0) * 0.025; // Reduced from 0.05
    float wave_y = cos(input_uv.x * 8.0 + time * 1.8) * 0.02;   // Reduced from 0.04
    
    centered.x += wave_x;
    centered.y += wave_y;
    
    return centered + 0.5;
}

/**
 * KALEIDOSCOPE FRACTAL VISION
 */
vec2 effect_kaleidoscope_fractal(vec2 input_uv, float time) {
    vec2 center = vec2(0.5);
    vec2 to_center = input_uv - center;
    
    float radius = length(to_center);
    float angle = atan(to_center.y, to_center.x);
    
    // Rotating kaleidoscope segments - reduced intensity
    float segments = 6.0 + sin(time * 0.5) * 1.0; // Reduced from 2.0
    angle += time * 0.4; // Reduced from 0.8
    
    float segment_angle = 6.28318 / segments;
    angle = mod(angle, segment_angle);
    
    // Mirror every other segment
    if (mod(floor(angle / segment_angle * segments), 2.0) == 1.0) {
        angle = segment_angle - angle;
    }
    
    // Fractal zoom - reduced intensity
    radius *= 1.0 + sin(time * 2.0) * 0.25; // Reduced from 0.5
    radius = fract(radius * 2.0); // Reduced from 3.0
    
    return center + vec2(cos(angle), sin(angle)) * radius;
}

/**
 * CHROMATIC SHIFT HALLUCINATION
 */
vec3 effect_chromatic_hallucination(vec3 base_color, vec2 uv_coords, float time) {
    // Separate RGB channels with different distortions - reduced intensity
    vec2 r_offset = vec2(sin(time * 3.0 + uv_coords.y * 20.0) * 0.01, 0.0);   // Reduced from 0.02
    vec2 g_offset = vec2(0.0, cos(time * 2.5 + uv_coords.x * 15.0) * 0.0075); // Reduced from 0.015
    vec2 b_offset = vec2(sin(time * 4.0) * 0.005, sin(time * 3.5) * 0.005);   // Reduced from 0.01
    
    float r = texture(tex, uv_coords + r_offset).r;
    float g = texture(tex, uv_coords + g_offset).g;
    float b = texture(tex, uv_coords + b_offset).b;
    
    return vec3(r, g, b);
}

/**
 * MORPHING GEOMETRY PATTERNS
 */
vec3 effect_morphing_patterns(vec3 base_color, vec2 uv_coords, float time) {
    vec2 grid = uv_coords * 8.0;
    vec2 grid_id = floor(grid);
    vec2 grid_uv = fract(grid);
    
    // Morphing between different geometric patterns
    float morph = sin(time * 1.2) * 0.5 + 0.5;
    
    // Pattern 1: Circles
    float circle = length(grid_uv - 0.5);
    circle = step(0.3, circle);
    
    // Pattern 2: Squares
    vec2 square_dist = abs(grid_uv - 0.5);
    float square = max(square_dist.x, square_dist.y);
    square = step(0.3, square);
    
    // Pattern 3: Triangles
    float triangle = abs(grid_uv.x - 0.5) + abs(grid_uv.y - 0.5);
    triangle = step(0.6, triangle);
    
    // Blend patterns based on time
    float pattern = mix(circle, square, morph);
    pattern = mix(pattern, triangle, sin(time * 0.8) * 0.5 + 0.5);
    
    // Color shift based on grid position
    vec3 pattern_color = vec3(
        sin(grid_id.x + time) * 0.5 + 0.5,
        cos(grid_id.y + time * 1.3) * 0.5 + 0.5,
        sin(grid_id.x + grid_id.y + time * 0.7) * 0.5 + 0.5
    );
    
    // Reduced blend factor for 50% opacity
    return mix(base_color, pattern_color, pattern * 0.3); // Reduced from 0.6
}

/**
 * LIQUID REALITY DISTORTION
 */
vec2 effect_liquid_distortion(vec2 input_uv, float time) {
    // Multiple layers of liquid-like distortion - reduced intensity
    float wave1 = sin(input_uv.x * 15.0 + time * 2.0) * sin(input_uv.y * 12.0 + time * 1.5);
    float wave2 = cos(input_uv.x * 8.0 + time * 1.8) * cos(input_uv.y * 10.0 + time * 2.2);
    float wave3 = sin(input_uv.x * 20.0 + input_uv.y * 18.0 + time * 3.0);
    
    vec2 distortion = vec2(
        wave1 * 0.015 + wave2 * 0.01,   // Reduced from 0.03 + 0.02
        wave2 * 0.0125 + wave3 * 0.0075 // Reduced from 0.025 + 0.015
    );
    
    return input_uv + distortion;
}

/**
 * INFINITE TUNNEL HALLUCINATION
 */
vec3 effect_infinite_tunnel(vec3 base_color, vec2 uv_coords, float time) {
    vec2 center = vec2(0.5);
    vec2 to_center = uv_coords - center;
    
    float radius = length(to_center);
    float angle = atan(to_center.y, to_center.x);
    
    // Infinite tunnel depth
    float tunnel_depth = fract(log(radius) + time * 0.25); // Reduced from 0.5
    
    // Spiral motion - reduced intensity
    angle += tunnel_depth * 5.0 + time * 1.0; // Reduced from 10.0 + 2.0
    
    // Tunnel rings
    float rings = sin(tunnel_depth * 50.0) * 0.5 + 0.5;
    
    // Psychedelic tunnel colors
    vec3 tunnel_color = vec3(
        sin(tunnel_depth * 10.0 + time * 2.0) * 0.5 + 0.5,
        cos(tunnel_depth * 12.0 + time * 2.5) * 0.5 + 0.5,
        sin(tunnel_depth * 8.0 + angle + time * 3.0) * 0.5 + 0.5
    );
    
    float intensity = rings * exp(-radius * 2.0);
    
    // Reduced blend factor for 50% opacity
    return mix(base_color, tunnel_color, intensity * 0.35); // Reduced from 0.7
}

/**
 * REALITY GLITCH EFFECT
 */
vec3 effect_reality_glitch(vec3 base_color, vec2 uv_coords, float time) {
    // Random glitch blocks
    vec2 block_size = vec2(20.0, 15.0);
    vec2 block_id = floor(uv_coords * block_size);
    
    // Pseudo-random function
    float random = fract(sin(dot(block_id, vec2(12.9898, 78.233)) + time * 0.1) * 43758.5453);
    
    // Glitch trigger - reduced chance
    float glitch_chance = 0.05; // Reduced from 0.1
    if (random < glitch_chance) {
        // Reduced intensity effects
        vec3 original = base_color;
        
        // Color inversion - partial
        base_color = mix(base_color, vec3(1.0) - base_color, 0.5);
        
        // Channel shifting - reduced
        base_color = mix(base_color, base_color.gbr, 0.3);
        
        // Brightness spike - reduced
        base_color *= 1.2 + sin(time * 20.0) * 0.2; // Reduced from 1.5 + 0.5
        
        // Blend with original for reduced effect
        base_color = mix(original, base_color, 0.6);
    }
    
    return base_color;
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
    vec2 working_uv = uv;
    vec2 original_uv = uv;
    
    // Apply hallucination distortions with reduced intensity
    working_uv = mix(original_uv, effect_breathing_walls(working_uv, u_time), 0.5);       // 50% blend
    working_uv = mix(working_uv, effect_liquid_distortion(working_uv, u_time), 0.3);     // 30% blend
    working_uv = mix(working_uv, effect_kaleidoscope_fractal(working_uv, u_time * 0.6), 0.4); // 40% blend
    
    // Sample texture with chromatic hallucination
    vec3 working_color = effect_chromatic_hallucination(vec3(0.0), working_uv, u_time);
    
    // Layer hallucination effects with reduced intensity
    working_color = effect_morphing_patterns(working_color, uv, u_time);
    working_color = effect_infinite_tunnel(working_color, uv, u_time);
    working_color = effect_reality_glitch(working_color, uv, u_time);
    
    // Get original texture for blending
    vec3 original_color = texture(tex, uv).rgb;
    
    // Reduce enhancement effects
    working_color = pow(working_color, vec3(0.9)); // Reduced from 0.8
    working_color *= 1.15; // Reduced from 1.3
    
    // Reduce psychedelic color cycling
    float color_cycle = sin(u_time * 1.5) * 0.1 + 1.0; // Reduced from 0.2
    working_color *= color_cycle;
    
    // Final 50% blend with original image
    working_color = mix(original_color, working_color, 0.5);
    
    // Add subtitle overlay
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}