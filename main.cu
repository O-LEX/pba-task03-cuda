#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <iomanip>
#include <string>
#include <cstdio>

#include <cuda_runtime.h>
#include "helper_math.h" // Include helper_math.h for CUDA math functions
#include "helper_image.h" // Include helper_image.h for image processing functions

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct Particle {
    float2 pos;
    float2 velocity;
    float2 force;

    Particle() : pos(make_float2(0.f, 0.f)), velocity(make_float2(0.f, 0.f)), force(make_float2(0.f, 0.f)) {}
};

// GPU kernel for brute force N-body simulation
__global__ void computeForcesKernel(float2* pos, float2* force, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    float2 myPos = pos[idx];
    float2 totalForce = make_float2(0.0f, 0.0f);
    
    // Gravitational constant and softening factor
    const float eps = 2.0e-3f;
    
    for (int j = 0; j < numParticles; j++) {
        if (idx == j) continue;
        
        float2 otherPos = pos[j];
        float2 diff = otherPos - myPos;
        
        float r_sq = dot(diff, diff) + eps * eps;
        float r = sqrtf(r_sq);
        if (r > 0.0f) {
            totalForce += diff * (1.0f / (r * r * r));
        }
    }
    
    force[idx] = totalForce;
}

// GPU kernel for updating particle positions and velocities
__global__ void updateParticlesKernel(float2* pos, float2* vel, float2* force, int numParticles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    vel[idx] += force[idx] * dt;    // Update velocity
    pos[idx] += vel[idx] * dt;      // Update position
}

// GPU kernel for cell-based acceleration calculation
__global__ void computeGridCellDataKernel(
    float2* pos, 
    int* cellIndex, 
    float2* cellCG, 
    int* cellCount, 
    int numParticles, 
    float boxSize,
    int numDiv) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    float2 particlePos = pos[idx];
    float h_inv = static_cast<float>(numDiv) / boxSize;
    int ix = static_cast<int>(floorf((particlePos.x + boxSize * 0.5f) * h_inv));
    int iy = static_cast<int>(floorf((particlePos.y + boxSize * 0.5f) * h_inv));
    
    int cellIdx = -1; // Default invalid
    if (ix >= 0 && ix < numDiv && iy >= 0 && iy < numDiv) {
        cellIdx = iy * numDiv + ix;
        cellIndex[idx] = cellIdx;
        
        // Atomically add to cell center of gravity and count
        atomicAdd(&cellCG[cellIdx].x, particlePos.x);
        atomicAdd(&cellCG[cellIdx].y, particlePos.y);
        atomicAdd(&cellCount[cellIdx], 1);
    } else {
        cellIndex[idx] = -1; // Particle is outside grid
    }
}

// GPU kernel to normalize center of gravity
__global__ void normalizeCGKernel(float2* cellCG, int* cellCount, int numCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCells) return;
    
    if (cellCount[idx] > 0) {
        cellCG[idx] /= static_cast<float>(cellCount[idx]);
    }
}

// GPU kernel for accelerated force computation - optimized version
__global__ void computeAcceleratedForcesKernel(
    float2* pos, 
    float2* force, 
    int* cellIndex,
    float2* cellCG, 
    int* cellParticleCount, 
    int numParticles,
    float boxSize,
    int numDiv) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    float2 myPos = pos[idx];
    float2 totalForce = make_float2(0.0f, 0.0f);
    const float eps = 2.0e-3f;
    
    // Calculate current cell
    float h_inv = static_cast<float>(numDiv) / boxSize;
    int ix = static_cast<int>(floorf((myPos.x + boxSize * 0.5f) * h_inv));
    int iy = static_cast<int>(floorf((myPos.y + boxSize * 0.5f) * h_inv));
    
    // Only process particles within the grid
    if (ix >= 0 && ix < numDiv && iy >= 0 && iy < numDiv) {
        // Process only neighboring cells (3x3 area)
        for (int dy = -1; dy <= 1; dy++) {
            int cy = iy + dy;
            if (cy < 0 || cy >= numDiv) continue;
            
            for (int dx = -1; dx <= 1; dx++) {
                int cx = ix + dx;
                if (cx < 0 || cx >= numDiv) continue;
                
                int cellIdx = cy * numDiv + cx;
                
                // Direct calculation with particles in near field
                for (int j = 0; j < numParticles; j++) {
                    if (idx == j) continue;
                    if (cellIndex[j] != cellIdx) continue; // Only particles in this cell
                    
                    float2 otherPos = pos[j];
                    float2 diff = otherPos - myPos;
                    
                    float r_sq = dot(diff, diff) + eps * eps;
                    float r = sqrtf(r_sq);
                    if (r > 0.0f) {
                        totalForce += diff * (1.0f / (r * r * r));
                    }
                }
            }
        }
        
        // Process far field cells in batch
        // Process all cells except the 3x3 neighboring area
        for (int cy = 0; cy < numDiv; cy++) {
            for (int cx = 0; cx < numDiv; cx++) {
                // Skip near field cells (already processed)
                if (abs(cx - ix) <= 1 && abs(cy - iy) <= 1) continue;
                
                int cellIdx = cy * numDiv + cx;
                
                // Only process cells that contain particles
                if (cellParticleCount[cellIdx] > 0) {
                    float2 cg = cellCG[cellIdx];
                    float2 diff = cg - myPos;
                    
                    float r_sq = dot(diff, diff) + eps * eps;
                    float r = sqrtf(r_sq);
                    if (r > 0.0f) {
                        totalForce += diff * (static_cast<float>(cellParticleCount[cellIdx]) / (r * r * r));
                    }
                }
            }
        }
    } else {
        // Particles outside grid are calculated using brute force
        for (int j = 0; j < numParticles; j++) {
            if (idx == j) continue;
            
            float2 otherPos = pos[j];
            float2 diff = otherPos - myPos;
            
            float r_sq = dot(diff, diff) + eps * eps;
            float r = sqrtf(r_sq);
            if (r > 0.0f) {
                totalForce += diff * (1.0f / (r * r * r));
            }
        }
    }
    
    force[idx] = totalForce;
}

// Helper function to save frame using helper_image.h - simplification to only use PPM format
void save_frame_ppm(const std::vector<Particle>& particles, int frame_idx, int img_size, float box_size_param) {
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
        float norm_y = (p.pos.y - world_min_y) / world_height; 

        // Convert to pixel coordinates
        int px = static_cast<int>(norm_x * img_size);
        int py = static_cast<int>((1.0f - norm_y) * img_size); // Invert Y for image coord system

        if (px >= 0 && px < img_size && py >= 0 && py < img_size) {
            int idx = (py * img_size + px) * 3;
            image_data[idx] = 0;   // R (black particle)
            image_data[idx+1] = 0; // G
            image_data[idx+2] = 0; // B
        }
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "frame_%03d.ppm", frame_idx);
    
    // Only save as PPM format for simplicity
    __savePPM(filename, image_data.data(), img_size, img_size, 3);
}

// Host function to run simulation with CUDA acceleration
void runSimulationCUDA(
    std::vector<Particle>& particles, 
    int numFrames, 
    float dt, 
    float boxSize,
    int numDiv,
    bool useAcceleration,
    int imgSize) {
    
    int numParticles = particles.size();
    
    // Allocate CUDA memory
    float2 *d_pos, *d_vel, *d_force, *d_cellCG;
    int *d_cellIndex, *d_cellCount;
    
    CUDA_CHECK(cudaMalloc(&d_pos, numParticles * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_vel, numParticles * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_force, numParticles * sizeof(float2)));
    
    // Prepare host data
    std::vector<float2> h_pos(numParticles);
    std::vector<float2> h_vel(numParticles);
    
    for (int i = 0; i < numParticles; i++) {
        h_pos[i] = particles[i].pos;
        h_vel[i] = particles[i].velocity;
    }
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), numParticles * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), numParticles * sizeof(float2), cudaMemcpyHostToDevice));
    
    // Grid-based acceleration data
    int numCells = numDiv * numDiv;
    
    if (useAcceleration) {
        CUDA_CHECK(cudaMalloc(&d_cellIndex, numParticles * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cellCG, numCells * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_cellCount, numCells * sizeof(int)));
    }
    
    // CUDA kernel configuration
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    int cellBlocks = (numCells + blockSize - 1) / blockSize;
    
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float forceComputeTime = 0.0f; // Time spent on force computation
    
    // Frame loop
    for (int frame = 0; frame < numFrames; frame++) {
        // Compute forces
        cudaEventRecord(start);
        
        if (useAcceleration) {
            // Reset cell data
            CUDA_CHECK(cudaMemset(d_cellCG, 0, numCells * sizeof(float2)));
            CUDA_CHECK(cudaMemset(d_cellCount, 0, numCells * sizeof(int)));
            
            // Step 1: Compute grid cell data
            computeGridCellDataKernel<<<numBlocks, blockSize>>>(
                d_pos, d_cellIndex, d_cellCG, d_cellCount, 
                numParticles, boxSize, numDiv);
            
            // Step 2: Normalize center of gravity
            normalizeCGKernel<<<cellBlocks, blockSize>>>(
                d_cellCG, d_cellCount, numCells);
            
            // Step 3: Compute forces with acceleration
            computeAcceleratedForcesKernel<<<numBlocks, blockSize>>>(
                d_pos, d_force, d_cellIndex, d_cellCG, d_cellCount,
                numParticles, boxSize, numDiv);
        } else {
            // Brute-force approach
            computeForcesKernel<<<numBlocks, blockSize>>>(
                d_pos, d_force, numParticles);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        forceComputeTime += milliseconds;
        
        // Update particles
        updateParticlesKernel<<<numBlocks, blockSize>>>(
            d_pos, d_vel, d_force, numParticles, dt);
        
        // Save frames at specified intervals
        if (frame % 20 == 0) {
            std::cout << "Simulating frame " << frame << " / " << numFrames 
                      << " Force compute time per frame: " << (forceComputeTime / (frame + 1)) << " ms" 
                      << std::endl;
            
            // Copy positions back to host for visualization
            CUDA_CHECK(cudaMemcpy(h_pos.data(), d_pos, numParticles * sizeof(float2), cudaMemcpyDeviceToHost));
            
            // Update particle positions for visualization
            for (int i = 0; i < numParticles; i++) {
                particles[i].pos = h_pos[i];
            }
            
            // Save the frame using helper_image.h
            save_frame_ppm(particles, frame, imgSize, boxSize);
        }
        
        // Check for CUDA errors
        CUDA_CHECK(cudaGetLastError());
    }
    
    std::cout << "Average force computation time: " << (forceComputeTime / numFrames) << " ms per frame"
              << " (" << (useAcceleration ? "accelerated" : "brute force") << ")" << std::endl;
    
    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free CUDA memory
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_force));
    
    if (useAcceleration) {
        CUDA_CHECK(cudaFree(d_cellIndex));
        CUDA_CHECK(cudaFree(d_cellCG));
        CUDA_CHECK(cudaFree(d_cellCount));
    }
}

struct Args {
    size_t num_particle = 1000;
    bool accelerate = false;
};

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
    std::cout << "CUDA Version: Accelerate: " << (args.accelerate ? "true" : "false")
              << ", Number of Particle: " << args.num_particle << std::endl;

    const float box_size_val = 1.5f;
    
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

    // Run CUDA simulation
    runSimulationCUDA(particles, num_total_frames, dt, box_size_val, num_div, args.accelerate, img_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "CUDA Computation time: " << std::fixed << std::setprecision(2) << diff.count() << "s" << std::endl;

    return 0;
}