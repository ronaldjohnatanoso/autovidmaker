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
// SIDE-VIEW BLACKHOLE EFFECT
// ============================================================================

/**
 * SIDE-VIEW BLACKHOLE WITH VERTICAL ACCRETION DISK
 */
vec3 effect_blackhole(vec2 uv_coords, float time) {
    // Moving blackhole center
    vec2 center = vec2(0.5 + sin(time * 0.2) * 0.1, 0.5 + cos(time * 0.15) * 0.05);
    vec2 to_center = uv_coords - center;
    float distance = length(to_center);
    
    // Event horizon - circular dark center
    float event_horizon = 0.08 + sin(time * 0.3) * 0.01;
    
    // Side-view accretion disk - flattened ellipse
    float disk_height = 0.15; // How tall the disk appears from side
    float disk_width = 0.4;   // How wide the disk extends
    
    // Calculate elliptical distance for side-view disk
    float ellipse_x = to_center.x / disk_width;
    float ellipse_y = to_center.y / disk_height;
    float ellipse_dist = sqrt(ellipse_x * ellipse_x + ellipse_y * ellipse_y);
    
    // Disk regions
    float disk_inner = 0.3;
    float disk_outer = 1.0;
    
    // Rotating motion - faster closer to center
    float rotation_angle = atan(to_center.y, to_center.x);
    rotation_angle += time * 2.0 / (distance + 0.1);
    
    // Disk brightness - only in the flattened elliptical region
    float disk_intensity = 0.0;
    if (ellipse_dist > disk_inner && ellipse_dist < disk_outer && distance > event_horizon) {
        // Bright inner edge, fading outward
        float disk_position = (ellipse_dist - disk_inner) / (disk_outer - disk_inner);
        disk_intensity = (1.0 - disk_position) * (1.0 - disk_position); // Quadratic falloff
        
        // Add turbulence and rotation effects
        float turbulence = sin(rotation_angle * 6.0 + time * 3.0) * cos(ellipse_dist * 15.0 + time * 2.0);
        disk_intensity += turbulence * 0.2 * disk_intensity;
        
        // Gravitational lensing - brighter on approaching side
        float lensing = 1.0 + sin(rotation_angle + time * 0.8) * 0.5;
        disk_intensity *= lensing;
        
        // Perspective dimming for side view
        float perspective_fade = exp(-abs(to_center.y) * 3.0);
        disk_intensity *= perspective_fade;
    }
    
    // Event horizon shadow - completely dark
    if (distance < event_horizon) {
        return vec3(0.0, 0.0, 0.0); // Pure black
    }
    
    // Photon sphere glow around event horizon
    float photon_glow = 0.0;
    if (distance < event_horizon * 2.0 && distance >= event_horizon) {
        photon_glow = exp(-(distance - event_horizon) * 15.0) * 0.4;
    }
    
    // Accretion disk colors - hot plasma with side-view perspective
    vec3 disk_color = vec3(0.0);
    if (disk_intensity > 0.0) {
        // Temperature gradient based on distance and intensity
        float temperature = disk_intensity;
        
        // Side-view color temperature - hotter in center, cooler at edges
        disk_color = vec3(
            1.0,                                    // Red - always present
            temperature * 0.7,                     // Orange/yellow
            temperature * temperature * 0.9       // Blue-white for hottest regions
        );
        
        disk_color *= disk_intensity;
        
        // Add relativistic beaming - brighter on approaching side
        float beaming = 1.0 + sin(rotation_angle + time * 0.8) * 0.3;
        disk_color *= beaming;
    }
    
    // Add photon sphere glow
    vec3 final_color = disk_color + vec3(photon_glow * 0.6, photon_glow * 0.4, photon_glow * 0.9);
    
    // Gravitational redshift - redder at edges
    float redshift = exp(-distance * 1.5) * 0.15;
    final_color.r += redshift;
    
    // Jets - vertical streams from poles (optional)
    float jet_distance = abs(to_center.x);
    if (jet_distance < 0.02 && abs(to_center.y) > event_horizon * 1.5) {
        float jet_intensity = exp(-jet_distance * 50.0) * 0.3;
        jet_intensity *= sin(abs(to_center.y) * 10.0 + time * 4.0) * 0.5 + 0.5;
        final_color += vec3(jet_intensity * 0.4, jet_intensity * 0.6, jet_intensity * 1.0);
    }
    
    return final_color;
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
    // Get original texture
    vec3 original_color = texture(tex, uv).rgb;
    
    // Create blackhole effect
    vec3 blackhole_overlay = effect_blackhole(uv, u_time);
    
    // Blend original with blackhole effect at 50% opacity
    vec3 final_color = mix(original_color, blackhole_overlay, 0.5);
    
    // Add subtitle overlay
    final_color = add_subtitle_overlay(final_color, uv);
    
    color = vec4(final_color, 1.0);
}