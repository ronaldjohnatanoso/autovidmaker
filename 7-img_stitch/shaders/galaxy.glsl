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
// 3-BODY PROBLEM ORANGE BLACKHOLES WITH MERGING
// ============================================================================

/**
 * ORANGE BLACKHOLE EFFECT WITH DYNAMIC SIZE
 */
vec3 effect_orange_blackhole(vec2 uv_coords, vec2 center, float time, float phase, float size_multiplier) {
    vec2 to_center = uv_coords - center;
    float distance = length(to_center);
    float angle = atan(to_center.y, to_center.x);
    
    // Event horizon with dynamic size
    float event_horizon = (0.03 + sin(time * 2.0 + phase) * 0.005) * size_multiplier;
    
    // Accretion disk
    float disk_inner = event_horizon * 1.5;
    float disk_outer = event_horizon * 6.0; // Made larger for more dramatic effect
    
    // Rotating disk - faster when excited
    angle += time * (3.0 + size_multiplier * 2.0) / (distance + 0.01);
    
    vec3 blackhole_color = vec3(0.0);
    
    // Event horizon - pure black
    if (distance < event_horizon) {
        return vec3(0.0);
    }
    
    // Accretion disk - orange plasma with intensity based on size
    if (distance > disk_inner && distance < disk_outer) {
        float disk_position = (distance - disk_inner) / (disk_outer - disk_inner);
        float disk_intensity = (1.0 - disk_position) * (1.0 - disk_position) * size_multiplier;
        
        // Add turbulence - more chaotic when merging
        float turbulence = sin(angle * 8.0 + time * 4.0 + phase) * cos(distance * 30.0 + time * 3.0);
        disk_intensity += turbulence * 0.4 * disk_intensity;
        
        // Orange color gradient - brighter when excited
        blackhole_color = vec3(
            1.0 * disk_intensity,        // Red
            0.6 * disk_intensity,        // Orange
            0.1 * disk_intensity * size_multiplier // Blue increases during merger
        );
    }
    
    // Photon sphere glow - enhanced during merger
    if (distance < disk_outer && distance >= event_horizon) {
        float glow = exp(-(distance - event_horizon) * 15.0) * 0.5 * size_multiplier;
        blackhole_color += vec3(glow * 0.9, glow * 0.6, glow * 0.2);
    }
    
    return blackhole_color;
}

/**
 * GRAVITATIONAL WAVE EFFECT
 */
vec3 effect_gravitational_waves(vec2 uv_coords, vec2 source1, vec2 source2, float time, float intensity) {
    vec2 wave_center = (source1 + source2) * 0.5;
    float wave_distance = length(uv_coords - wave_center);
    
    // Multiple wave frequencies
    float wave1 = sin(wave_distance * 15.0 - time * 8.0) * intensity;
    float wave2 = sin(wave_distance * 25.0 - time * 12.0) * intensity * 0.7;
    float wave3 = sin(wave_distance * 35.0 - time * 16.0) * intensity * 0.5;
    
    float combined_waves = (wave1 + wave2 + wave3) * exp(-wave_distance * 2.0);
    
    // Wave colors - bluish distortion
    return vec3(combined_waves * 0.1, combined_waves * 0.2, combined_waves * 0.4);
}

/**
 * MERGER EXPLOSION EFFECT
 */
vec3 effect_merger_explosion(vec2 uv_coords, vec2 merger_point, float explosion_time, float max_time) {
    if (explosion_time <= 0.0 || explosion_time > max_time) {
        return vec3(0.0);
    }
    
    float distance = length(uv_coords - merger_point);
    float explosion_progress = explosion_time / max_time;
    
    // Expanding shockwave
    float shockwave_radius = explosion_progress * 0.8;
    float shockwave_thickness = 0.05;
    float shockwave_intensity = 0.0;
    
    if (abs(distance - shockwave_radius) < shockwave_thickness) {
        shockwave_intensity = (1.0 - explosion_progress) * 
                             exp(-abs(distance - shockwave_radius) / shockwave_thickness);
    }
    
    // Energy burst
    float energy_burst = exp(-distance * 8.0) * (1.0 - explosion_progress) * 
                        sin(explosion_time * 20.0) * 0.5 + 0.5;
    
    // Plasma jets
    vec2 to_center = uv_coords - merger_point;
    float jet_alignment = abs(dot(normalize(to_center), vec2(0, 1)));
    float jet_intensity = 0.0;
    if (jet_alignment > 0.9 && distance > 0.05) {
        jet_intensity = exp(-distance * 5.0) * (1.0 - explosion_progress) * 0.8;
    }
    
    // Combined explosion colors
    vec3 explosion_color = vec3(
        (shockwave_intensity + energy_burst) * 1.2,  // Bright white-orange
        (shockwave_intensity + energy_burst) * 0.8,  // Orange
        (shockwave_intensity + jet_intensity) * 0.6  // Blue jets
    );
    
    return explosion_color;
}

/**
 * 3-BODY ORBITAL MECHANICS WITH MERGING
 */
vec3 effect_three_body_system(vec2 uv_coords, float time) {
    // 3-body problem orbital parameters
    float orbit_speed = time * 0.6; // Slower for more drama
    
    // Body 1 - Large orbit
    vec2 body1_center = vec2(
        0.5 + cos(orbit_speed) * 0.25,
        0.5 + sin(orbit_speed) * 0.2
    );
    
    // Body 2 - Counter-rotating, different speed
    vec2 body2_center = vec2(
        0.5 + cos(-orbit_speed * 1.3 + 2.094) * 0.2,
        0.5 + sin(-orbit_speed * 1.3 + 2.094) * 0.25
    );
    
    // Body 3 - Complex chaotic orbit
    float chaos_factor = sin(orbit_speed * 0.7) * 0.08;
    vec2 body3_center = vec2(
        0.5 + cos(orbit_speed * 0.9 + 4.188) * (0.15 + chaos_factor),
        0.5 + sin(orbit_speed * 0.9 + 4.188) * (0.18 + chaos_factor)
    );
    
    // Calculate distances for collision detection
    float dist12 = length(body1_center - body2_center);
    float dist13 = length(body1_center - body3_center);
    float dist23 = length(body2_center - body3_center);
    
    // Collision threshold
    float collision_threshold = 0.08;
    
    // Check for active mergers
    bool merger12 = dist12 < collision_threshold;
    bool merger13 = dist13 < collision_threshold;
    bool merger23 = dist23 < collision_threshold;
    
    // Calculate size multipliers based on interactions
    float size1 = 1.0 + (merger12 ? (collision_threshold - dist12) * 5.0 : 0.0) +
                       (merger13 ? (collision_threshold - dist13) * 5.0 : 0.0);
    float size2 = 1.0 + (merger12 ? (collision_threshold - dist12) * 5.0 : 0.0) +
                       (merger23 ? (collision_threshold - dist23) * 5.0 : 0.0);
    float size3 = 1.0 + (merger13 ? (collision_threshold - dist13) * 5.0 : 0.0) +
                       (merger23 ? (collision_threshold - dist23) * 5.0 : 0.0);
    
    // Create each blackhole with dynamic sizing
    vec3 blackhole1 = effect_orange_blackhole(uv_coords, body1_center, time, 0.0, size1);
    vec3 blackhole2 = effect_orange_blackhole(uv_coords, body2_center, time, 2.094, size2);
    vec3 blackhole3 = effect_orange_blackhole(uv_coords, body3_center, time, 4.188, size3);
    
    // Gravitational waves during close encounters
    vec3 grav_waves = vec3(0.0);
    if (dist12 < 0.2) {
        float wave_intensity = (0.2 - dist12) * 2.0;
        grav_waves += effect_gravitational_waves(uv_coords, body1_center, body2_center, time, wave_intensity);
    }
    if (dist13 < 0.2) {
        float wave_intensity = (0.2 - dist13) * 2.0;
        grav_waves += effect_gravitational_waves(uv_coords, body1_center, body3_center, time, wave_intensity);
    }
    if (dist23 < 0.2) {
        float wave_intensity = (0.2 - dist23) * 2.0;
        grav_waves += effect_gravitational_waves(uv_coords, body2_center, body3_center, time, wave_intensity);
    }
    
    // Merger explosions
    vec3 explosions = vec3(0.0);
    if (merger12) {
        vec2 merger_point = (body1_center + body2_center) * 0.5;
        float explosion_time = (collision_threshold - dist12) * 20.0; // Scale explosion time
        explosions += effect_merger_explosion(uv_coords, merger_point, explosion_time, 3.0);
    }
    if (merger13) {
        vec2 merger_point = (body1_center + body3_center) * 0.5;
        float explosion_time = (collision_threshold - dist13) * 20.0;
        explosions += effect_merger_explosion(uv_coords, merger_point, explosion_time, 3.0);
    }
    if (merger23) {
        vec2 merger_point = (body2_center + body3_center) * 0.5;
        float explosion_time = (collision_threshold - dist23) * 20.0;
        explosions += effect_merger_explosion(uv_coords, merger_point, explosion_time, 3.0);
    }
    
    // Enhanced orbital trails with fading
    vec3 trails = vec3(0.0);
    for (float i = 0.0; i < 15.0; i += 1.0) {
        float trail_time = time - i * 0.08;
        vec2 trail1 = vec2(
            0.5 + cos(trail_time * 0.6) * 0.25,
            0.5 + sin(trail_time * 0.6) * 0.2
        );
        vec2 trail2 = vec2(
            0.5 + cos(-trail_time * 0.6 * 1.3 + 2.094) * 0.2,
            0.5 + sin(-trail_time * 0.6 * 1.3 + 2.094) * 0.25
        );
        vec2 trail3 = vec2(
            0.5 + cos(trail_time * 0.6 * 0.9 + 4.188) * 0.15,
            0.5 + sin(trail_time * 0.6 * 0.9 + 4.188) * 0.18
        );
        
        float trail_fade = (15.0 - i) / 15.0 * 0.08;
        trails += vec3(trail_fade * 0.4, trail_fade * 0.25, trail_fade * 0.1) * 
                  (exp(-length(uv_coords - trail1) * 120.0) + 
                   exp(-length(uv_coords - trail2) * 120.0) + 
                   exp(-length(uv_coords - trail3) * 120.0));
    }
    
    return blackhole1 + blackhole2 + blackhole3 + grav_waves + explosions + trails;
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
    
    // Create 3-body blackhole system with merging
    vec3 blackhole_system = effect_three_body_system(uv, u_time);
    
    // Blend original with blackhole system at 50% opacity
    vec3 final_color = mix(original_color, blackhole_system, 0.5);
    
    // Add subtitle overlay
    final_color = add_subtitle_overlay(final_color, uv);
    
    color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
}