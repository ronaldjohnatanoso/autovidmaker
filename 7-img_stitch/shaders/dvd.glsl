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
// GAUSSIAN SPLATTING EFFECT
// ============================================================================

// Generate pseudo-random splat positions and properties
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 hash3(vec2 p) {
    vec3 q = vec3(dot(p, vec2(127.1, 311.7)), 
                  dot(p, vec2(269.5, 183.3)), 
                  dot(p, vec2(419.2, 371.9)));
    return fract(sin(q) * 43758.5453);
}

vec3 gaussian_splatting(vec2 uv_coords, float time) {
    vec3 final_color = vec3(0.0);
    float total_weight = 0.0;
    
    // Number of splats to render
    int num_splats = 20;
    
    for(int i = 0; i < num_splats; i++) {
        // Generate splat properties based on index and time
        vec2 seed = vec2(float(i) * 0.1, time * 0.1);
        vec3 rand = hash3(seed);
        
        // Splat center position (animated)
        vec2 splat_center = vec2(
            0.5 + sin(time * 0.5 + float(i) * 0.3) * 0.4,
            0.5 + cos(time * 0.3 + float(i) * 0.7) * 0.4
        );
        
        // Distance from current pixel to splat center
        vec2 delta = uv_coords - splat_center;
        
        // Splat size (varies with time and index)
        float splat_size = 0.05 + sin(time + float(i)) * 0.03;
        
        // Gaussian falloff
        float gaussian = exp(-dot(delta, delta) / (2.0 * splat_size * splat_size));
        
        // Sample original texture at splat center with some offset
        vec2 sample_uv = splat_center + sin(time + float(i)) * 0.02;
        sample_uv = clamp(sample_uv, 0.0, 1.0);
        vec3 splat_color = texture(tex, sample_uv).rgb;
        
        // Color modulation based on splat properties
        splat_color *= 1.0 + sin(time * 2.0 + float(i) * 0.5) * 0.3;
        
        // Accumulate color weighted by Gaussian
        final_color += splat_color * gaussian;
        total_weight += gaussian;
    }
    
    // Normalize and add some base color from original texture
    if(total_weight > 0.001) {
        final_color /= total_weight;
    }
    
    // Blend with original texture for background
    vec3 background = texture(tex, uv_coords).rgb * 0.3;
    return mix(background, final_color, smoothstep(0.0, 0.1, total_weight));
}

// ============================================================================
// 3D GAUSSIAN SPLATTING (More advanced)
// ============================================================================

vec3 gaussian_splatting_3d(vec2 uv_coords, float time) {
    vec3 final_color = vec3(0.0);
    float total_alpha = 0.0;
    
    // Camera setup
    vec3 camera_pos = vec3(0.0, 0.0, -3.0);
    vec3 ray_dir = normalize(vec3((uv_coords - 0.5) * 2.0, 1.0));
    
    // Number of 3D splats
    int num_splats = 15;
    
    for(int i = 0; i < num_splats; i++) {
        float fi = float(i);
        
        // 3D splat position (animated)
        vec3 splat_pos = vec3(
            sin(time * 0.3 + fi * 0.8) * 2.0,
            cos(time * 0.4 + fi * 0.6) * 1.5,
            sin(time * 0.2 + fi * 0.4) * 1.0
        );
        
        // Project 3D position to screen
        vec3 to_splat = splat_pos - camera_pos;
        float depth = dot(to_splat, vec3(0, 0, 1));
        
        if(depth > 0.1) {
            vec2 screen_pos = to_splat.xy / depth + 0.5;
            
            // Distance from current pixel to projected splat
            vec2 delta = uv_coords - screen_pos;
            
            // 3D Gaussian with depth-based size
            float splat_size = 0.1 / depth;
            float gaussian = exp(-dot(delta, delta) / (2.0 * splat_size * splat_size));
            
            // Sample texture with animated offset
            vec2 tex_uv = screen_pos + sin(time + fi) * 0.05;
            tex_uv = clamp(tex_uv, 0.0, 1.0);
            vec3 splat_color = texture(tex, tex_uv).rgb;
            
            // Depth-based color modulation
            splat_color *= (1.0 + sin(time * 3.0 + fi)) * 0.5;
            splat_color *= 1.0 / (1.0 + depth * 0.5); // Depth falloff
            
            // Alpha blending
            float alpha = gaussian * 0.8;
            final_color = final_color * (1.0 - alpha) + splat_color * alpha;
            total_alpha += alpha * (1.0 - total_alpha);
        }
    }
    
    // Background
    vec3 background = texture(tex, uv_coords).rgb;
    return mix(background, final_color, total_alpha);
}

// ============================================================================
// FIXED GAUSSIAN SPLATTING (WORKS BETTER)
// ============================================================================

vec3 gaussian_splatting_fixed(vec2 uv_coords, float time) {
    vec3 final_color = vec3(0.0);
    float total_alpha = 0.0;
    
    // Number of splats
    int num_splats = 25;
    
    for(int i = 0; i < num_splats; i++) {
        float fi = float(i);
        
        // Simple 2D splat positions that actually work
        vec2 splat_center = vec2(
            0.5 + sin(time * 0.8 + fi * 2.3) * 0.6,
            0.5 + cos(time * 0.6 + fi * 1.7) * 0.6
        );
        
        // Wrap around screen edges
        splat_center = fract(splat_center);
        
        // Distance from current pixel to splat center
        vec2 delta = uv_coords - splat_center;
        
        // Handle screen wrapping for seamless effect
        if(abs(delta.x) > 0.5) delta.x = sign(delta.x) * (1.0 - abs(delta.x));
        if(abs(delta.y) > 0.5) delta.y = sign(delta.y) * (1.0 - abs(delta.y));
        
        // Animated splat size
        float splat_size = 0.08 + sin(time * 2.0 + fi) * 0.04;
        
        // Gaussian blob
        float gaussian = exp(-dot(delta, delta) / (splat_size * splat_size));
        
        // Sample original texture at splat position
        vec2 tex_uv = splat_center + sin(time + fi) * 0.03;
        tex_uv = fract(tex_uv); // Keep in bounds
        vec3 splat_color = texture(tex, tex_uv).rgb;
        
        // Make splats more vibrant
        splat_color *= 1.5 + sin(time * 3.0 + fi) * 0.5;
        
        // Alpha blending
        float alpha = gaussian * 0.9;
        final_color = mix(final_color, splat_color, alpha);
        total_alpha = max(total_alpha, alpha);
    }
    
    // Background from original texture
    vec3 background = texture(tex, uv_coords).rgb * 0.4;
    return mix(background, final_color, total_alpha);
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
    // Use the fixed Gaussian splatting
    vec3 working_color = gaussian_splatting_fixed(uv, u_time);
    
    // Add subtitles
    working_color = add_subtitle_overlay(working_color, uv);
    
    color = vec4(working_color, 1.0);
}