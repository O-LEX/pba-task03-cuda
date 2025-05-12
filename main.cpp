#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // For std::sort, std::fill
#include <random>    // For random number generation
#include <chrono>    // For timing
#include <limits>    // For std::numeric_limits
#include <iomanip>   // For std::fixed, std::setprecision
#include <string>    // For std::string, std::stoul
#include <cstdio>    // For snprintf

#include "helper_math.h" // Include helper_math.h

// stb_image_write.h is available, so let's include it.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Particle {
    float2 pos; // Changed from Vec2f
    float2 velocity; // Changed from Vec2f
    float2 force; // Changed from Vec2f

    Particle() : pos(make_float2(0.f, 0.f)), velocity(make_float2(0.f, 0.f)), force(make_float2(0.f, 0.f)) {} // Default constructor
};

/// Gravitational force with softening
float2 gravitational_force(const float2& d) { // Changed from Vec2f
    const float eps = 2.0e-3f; // softening coefficient
    float r_sq = dot(d, d) + eps * eps; // Changed from d.norm_squared()
    float r = std::sqrt(r_sq);
    if (r == 0.f) return make_float2(0.f, 0.f);
    return d * (1.f / (r * r * r)); // Changed from d.scale()
}

/// For each particle, set summation of gravitational forces
/// from all the other particles in a brute-force way O(N^2)
void set_force_bruteforce(std::vector<Particle>& particles) {
    for (size_t ip = 0; ip < particles.size(); ++ip) {
        particles[ip].force = make_float2(0.f, 0.f); // Changed from Vec2f
        for (size_t jp = 0; jp < particles.size(); ++jp) {
            if (ip == jp) {
                continue;
            }
            particles[ip].force += gravitational_force(particles[jp].pos - particles[ip].pos);
        }
    }
}

/// position to grid coordinate
size_t cell_index_from_position(const float2& pos, float box_size, size_t num_div) { // Changed from Vec2f
    float h_inv = static_cast<float>(num_div) / box_size;
    long ix = static_cast<long>(std::floor((pos.x + box_size * 0.5f) * h_inv));
    long iy = static_cast<long>(std::floor((pos.y + box_size * 0.5f) * h_inv));

    if (ix < 0 || ix >= static_cast<long>(num_div)) {
        return std::numeric_limits<size_t>::max();
    }
    if (iy < 0 || iy >= static_cast<long>(num_div)) {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(iy * num_div + ix);
}

struct Acceleration {
    float box_size;
    size_t num_div;

    std::vector<size_t> cell2idx;
    std::vector<std::pair<size_t, size_t>> idx2ipic; // (particle_index, cell_index)
    std::vector<float2> cell2cg; // Changed from Vec2f

    Acceleration(float bs, size_t nd) : box_size(bs), num_div(nd) {}

    void construct(const std::vector<Particle>& particles) {
        size_t num_cell = num_div * num_div;
        idx2ipic.assign(particles.size(), {0,0}); // Pre-allocate then fill

        for (size_t i_particle = 0; i_particle < particles.size(); ++i_particle) {
            size_t i_grid = cell_index_from_position(particles[i_particle].pos, box_size, num_div);
            idx2ipic[i_particle] = {i_particle, i_grid};
        }

        std::sort(idx2ipic.begin(), idx2ipic.end(), [](const auto& a, const auto& b) {
            return a.second < b.second; // Sort by cell_index
        });

        cell2idx.assign(num_cell + 1, 0); 
        for (const auto& p_pair : idx2ipic) {
            size_t i_cell = p_pair.second;
            if (i_cell == std::numeric_limits<size_t>::max()) {
                continue;
            }
            if (i_cell < num_cell) { 
                 cell2idx[i_cell + 1]++;
            }
        }

        for (size_t i_grid = 0; i_grid < num_cell; ++i_grid) {
            cell2idx[i_grid + 1] += cell2idx[i_grid];
        }

        cell2cg.assign(num_cell, make_float2(0.f, 0.f)); // Changed from Vec2f

        for (size_t i_grid = 0; i_grid < num_cell; ++i_grid) {
            for (size_t idx = cell2idx[i_grid]; idx < cell2idx[i_grid + 1]; ++idx) {
                if (idx < idx2ipic.size()) { 
                    size_t i_particle = idx2ipic[idx].first;
                    if (idx2ipic[idx].second == i_grid) {
                         cell2cg[i_grid] += particles[i_particle].pos;
                    }
                }
            }
            size_t num_particle_in_cell = cell2idx[i_grid + 1] - cell2idx[i_grid];
            if (num_particle_in_cell > 0) {
                cell2cg[i_grid] /= static_cast<float>(num_particle_in_cell);
            }
        }
    }
};


void set_force_accelerated(std::vector<Particle>& particles, const Acceleration& acc) {
    for (size_t i_particle = 0; i_particle < particles.size(); ++i_particle) {
        size_t i_cell = cell_index_from_position(particles[i_particle].pos, acc.box_size, acc.num_div);
        particles[i_particle].force = make_float2(0.f, 0.f); // Changed from Vec2f

        size_t ix = acc.num_div; // Default for out-of-bounds, ensures far-field
        size_t iy = acc.num_div; 
        if (i_cell != std::numeric_limits<size_t>::max()) {
            ix = i_cell % acc.num_div;
            iy = i_cell / acc.num_div;
        }

        for (size_t j_cell = 0; j_cell < acc.num_div * acc.num_div; ++j_cell) {
            size_t jx = j_cell % acc.num_div;
            size_t jy = j_cell / acc.num_div;

            bool is_near_field = false;
            if (i_cell != std::numeric_limits<size_t>::max()) { // Only consider near field if current particle is in a valid cell
                 is_near_field = (std::abs(static_cast<long>(ix) - static_cast<long>(jx)) <= 1 &&
                                  std::abs(static_cast<long>(iy) - static_cast<long>(jy)) <= 1);
            }

            if (is_near_field) {
                for (size_t jdx = acc.cell2idx[j_cell]; jdx < acc.cell2idx[j_cell + 1]; ++jdx) {
                     if (jdx < acc.idx2ipic.size()) { 
                        size_t j_particle = acc.idx2ipic[jdx].first;
                        if (i_particle == j_particle) {
                            continue;
                        }
                        float2 diff = particles[j_particle].pos - particles[i_particle].pos; // Changed from Vec2f
                        particles[i_particle].force += gravitational_force(diff);
                    }
                }
            } else {
                // Far field approximation
                if (j_cell < acc.cell2cg.size()) { 
                    const float2& cg = acc.cell2cg[j_cell]; // Changed from Vec2f
                    size_t num_particle_in_cell = acc.cell2idx[j_cell + 1] - acc.cell2idx[j_cell];
                    if (num_particle_in_cell > 0) {
                         particles[i_particle].force += static_cast<float>(num_particle_in_cell) *
                                                       gravitational_force(cg - particles[i_particle].pos);
                    }
                }
            }
        }
    }
}

struct Args {
    size_t num_particle = 1000;
    bool accelerate = false;
};

void save_frame_png(const std::vector<Particle>& particles, int frame_idx, int img_size, float box_size_param) {
    std::vector<unsigned char> image_data(img_size * img_size * 3);
    // Fill background with white
    std::fill(image_data.begin(), image_data.end(), 255);

    float half_box = box_size_param * 0.5f;
    float world_min_x = -half_box;
    float world_max_x = half_box;
    float world_min_y = -half_box;
    float world_max_y = half_box;
    float world_width = world_max_x - world_min_x;
    float world_height = world_max_y - world_min_y;

    for (const auto& p : particles) {
        // Normalize coordinates to [0, 1] range
        float norm_x = (p.pos.x - world_min_x) / world_width;
        float norm_y = (p.pos.y - world_min_y) / world_height; // Y might be inverted depending on image coords

        // Convert to pixel coordinates
        int px = static_cast<int>(norm_x * img_size);
        int py = static_cast<int>((1.0f - norm_y) * img_size); // Invert Y for typical image origin (top-left)

        if (px >= 0 && px < img_size && py >= 0 && py < img_size) {
            int idx = (py * img_size + px) * 3;
            image_data[idx] = 0;   // R (black particle)
            image_data[idx+1] = 0; // G
            image_data[idx+2] = 0; // B
        }
    }
    char filename[256];
    snprintf(filename, sizeof(filename), "frame_%03d.png", frame_idx); // Changed filename format
    stbi_write_png(filename, img_size, img_size, 3, image_data.data(), img_size * 3);
}

int main(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg_str = argv[i];
        if (arg_str == "--num-particle" && i + 1 < argc) {
            try {
                args.num_particle = std::stoul(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid value for --num-particle: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg_str == "--accelerate") {
            args.accelerate = true;
        } else {
            std::cerr << "Unknown argument: " << arg_str << std::endl;
            std::cerr << "Usage: " << argv[0] << " [--num-particle N] [--accelerate]" << std::endl;
            return 1;
        }
    }

    const size_t num_div = 32;
    std::cout << "Accelerate: " << (args.accelerate ? "true" : "false")
              << ", Number of Particle: " << args.num_particle << std::endl;

    const float box_size_val = 1.5f;
    Acceleration acc(box_size_val, num_div);

    float aabb2[] = {
        -box_size_val * 0.5f, -box_size_val * 0.5f,
         box_size_val * 0.5f,  box_size_val * 0.5f
    };

    std::vector<Particle> particles(args.num_particle);
    std::mt19937 rng(std::random_device{}()); 
    std::uniform_real_distribution<float> dist_rand(0.0f, 1.0f);

    for (auto& p : particles) {
        p.pos.x = aabb2[0] + (aabb2[2] - aabb2[0]) * dist_rand(rng);
        p.pos.y = aabb2[1] + (aabb2[3] - aabb2[1]) * dist_rand(rng);
        float center_x = (aabb2[0] + aabb2[2]) * 0.5f;
        float center_y = (aabb2[1] + aabb2[3]) * 0.5f;
        p.velocity.x = 100.f * (p.pos.y - center_y);
        p.velocity.y = -100.f * (p.pos.x - center_x);
    }

    const int img_size = 300; 

    auto start_time = std::chrono::high_resolution_clock::now();
    const float dt = 0.00002f; 
    const int num_total_frames = 1000; 
    const int save_every_n_frames = 20;

    for (int i_frame = 0; i_frame < num_total_frames; ++i_frame) {
        if (args.accelerate) {
            acc.construct(particles);
            set_force_accelerated(particles, acc);
        } else {
            set_force_bruteforce(particles);
        }

        for (auto& p : particles) {
            p.velocity += p.force * dt;
            p.pos += p.velocity * dt;
        }

        if (i_frame % save_every_n_frames == 0) {
            std::cout << "Simulating frame " << i_frame << " / " << num_total_frames << std::endl;
            save_frame_png(particles, i_frame, img_size, box_size_val); // Pass i_frame directly
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Computation time: " << std::fixed << std::setprecision(2) << diff.count() << "s" << std::endl;

    return 0;
}
