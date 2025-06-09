#version 330
uniform sampler2D tex;        // The image texture
uniform float u_time;         // Time for animations
in vec2 uv;                   // UV coordinates from vertex shader
out vec4 color;               // Final pixel color

// ============================================================================
// INDEPENDENT EFFECT MODULES
// ============================================================================

/**
 * ZOOM MODULE - Completely self-contained
 * Input: any UV coordinates
 * Output: zoomed UV coordinates
 */
vec2 effect_zoom(vec2 input_uv, float time) {
    vec2 center = vec2(0.5, 0.5);
    float zoom_factor = 1.0 + sin(time * 2.0) * 0.3; // speed=2.0, intensity=0.3
    return center + (input_uv - center) / zoom_factor;
}

/**
 * SCANLINE NOISE MODULE - Completely independent color effect
 * Input: base color
 * Output: color with noise added
 */
vec3 effect_scanlines(vec3 input_color, vec2 uv_coords, float time) {
    float noise = sin(uv_coords.y * 1200.0 + time * 10.0) * 0.02;
    return input_color + noise;
}

/**
 * VIGNETTE MODULE - Completely independent color effect
 * Input: base color
 * Output: color with vignette applied
 */
vec3 effect_vignette(vec3 input_color, vec2 uv_coords) {
    vec2 vignette_center = uv_coords - 0.5;
    float vignette_factor = 1.0 - dot(vignette_center, vignette_center) * 0.9;
    return input_color * vignette_factor;
}

/**
 * DESATURATION MODULE - Completely independent color effect
 * Input: color
 * Output: desaturated color
 */
vec3 effect_desaturate(vec3 input_color, float amount) {
    float gray = dot(input_color, vec3(0.299, 0.587, 0.114));
    return mix(input_color, vec3(gray), amount);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
void main() {
    // Start with original UV coordinates
    vec2 working_uv = uv;
    
    // Apply UV transformations (comment to disable)
    working_uv = effect_zoom(working_uv, u_time);
    
    // Sample texture with final UV coordinates
    vec4 base_texture = texture(tex, working_uv);
    vec3 working_color = base_texture.rgb;
    
    // Apply color effects (comment to disable)
    working_color = effect_scanlines(working_color, uv, u_time);
    working_color = effect_vignette(working_color, uv);
    working_color = effect_desaturate(working_color, 0.1);
    
    // Output final result
    color = vec4(working_color, base_texture.a);
}