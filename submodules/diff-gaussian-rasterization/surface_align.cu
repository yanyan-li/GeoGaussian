#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <memory>
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "cuda_rasterizer/auxiliary.h"


#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <glm/glm.hpp>

#include "surface_align.h"

#define THETA_THRESHOLD 0.004       // 5 degree

std::function<char*(size_t N)> resizeFunctional_(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

BinningState BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain_(chunk, binning.point_list, P, 128);
	obtain_(chunk, binning.point_list_unsorted, P, 128);
	obtain_(chunk, binning.point_list_keys, P, 128);
	obtain_(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain_(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

__forceinline__ __device__ glm::mat3 buildRotation(const float4& q)
{
    float norm = sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);

    float4 q_ = {q.x / norm, q.y / norm, q.z / norm, q.w / norm};

    float r = q_.x;
    float x = q_.y;
    float y = q_.z;
    float z = q_.w;

	glm::mat3 R = glm::mat3(
		1 - 2 * (y*y + z*z), 2 * (x*y - r*z), 2 * (x*z + r*y),
		2 * (x*y + r*z), 1 - 2 * (x*x + z*z), 2 * (y*z - r*x),
		2 * (x*z - r*y), 2 * (y*z + r*x), 1 - 2 * (x*x + y*y));

    return R;
}

__global__ void initIndexCUDA(int P, uint32_t* point_list_unsorted)
{
   	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	point_list_unsorted[idx] = (uint32_t)idx;
}

__global__ void processCUDA(
        int P,
        const int* point_list_keys,
        const uint32_t* point_list,
        const float* xyzs,
        const int* xyz_ids,
        const float* rotations,
        float* out_loss_d,
        float* out_loss_normal)
{
   	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    int currentId = point_list_keys[idx];
    int nextId = (idx == P-1) ? -1 : point_list_keys[idx+1];

    if (currentId == -1 || currentId != nextId)
        return;

    float3 xyz_current = { xyzs[3*point_list[idx]], xyzs[3*point_list[idx] + 1], xyzs[3*point_list[idx] + 2] };
    float3 xyz_next = { xyzs[3*point_list[idx+1]], xyzs[3*point_list[idx+1] + 1], xyzs[3*point_list[idx+1] + 2] };
    float4 rotation_current = { rotations[4*point_list[idx]], rotations[4*point_list[idx] + 1], rotations[4*point_list[idx] + 2],  rotations[4*point_list[idx] + 3] };
    float4 rotation_next = { rotations[4*point_list[idx+1]], rotations[4*point_list[idx+1] + 1], rotations[4*point_list[idx+1] + 2], rotations[4*point_list[idx+1] + 3] };


    glm::mat3 R_current = buildRotation(rotation_current);
    glm::mat3 R_next = buildRotation(rotation_next);

    float3 normal_current = {R_current[0][2], R_current[1][2], R_current[2][2]};
    float3 normal_next = {R_next[0][2], R_next[1][2], R_next[2][2]};

    float d_current = xyz_current.x*normal_current.x + xyz_current.y*normal_current.y + xyz_current.z*normal_current.z;
    float d_next = xyz_next.x*normal_next.x + xyz_next.y*normal_next.y + xyz_next.z*normal_next.z;

    float d_mean = (d_current + d_next) / 2.0;
    float loss_d = abs(d_current - d_mean) + abs(d_next - d_mean);
    out_loss_d[point_list[idx]] = loss_d;
//     if (idx < 10)
//     {
//         printf("out loss d %f\n", out_loss_d[point_list[idx]]);
//     }
    float cos_theta = normal_current.x*normal_next.x + normal_current.y*normal_next.y + normal_current.z*normal_next.z;
    if (cos_theta <= 1)
    {
        out_loss_normal[point_list[idx]] = 1 - cos_theta;
//         if (idx < 10)
//         {
//             printf("out loss normal %f\n", out_loss_normal[point_list[idx]]);
//         }
    }
//     else
//     {
//          if (idx < 10)
//         {
//             printf("cos_theta %f\n", cos_theta);
//         }
//     }
}


__global__ void processKNNCUDA(
        int P,
        int K,
        const float* xyzs,
        const int* xyz_ids,
        const float* rotations,
        const int* indexs,
        float* mean_ds,
        float* out_loss_d,
        float* out_loss_normal)
{
   	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    float3 xyz_current = {xyzs[3*indexs[K*idx]], xyzs[3*indexs[K*idx] + 1], xyzs[3*indexs[K*idx] + 2]};
    float4 rotation_current = {rotations[4*indexs[K*idx]], rotations[4*indexs[K*idx] + 1], rotations[4*indexs[K*idx] + 2], rotations[4*indexs[K*idx] + 3]};
    float q_r = rotation_current.x;
    float q_x = rotation_current.y;
    float q_y = rotation_current.z;
    float q_z = rotation_current.w;
    float3 normal_current = {2 * (q_x*q_z + q_r*q_y), 2 * (q_y*q_z - q_r*q_x), 1 - 2 * (q_x*q_x + q_y*q_y)};

    for (int i = 0; i < K; i++)
    {
        float3 xyz = {xyzs[3*indexs[K*idx+i]], xyzs[3*indexs[K*idx+i] + 1], xyzs[3*indexs[K*idx+i] + 2]};
        float4 rotation = {rotations[4*indexs[K*idx+i]], rotations[4*indexs[K*idx+i] + 1], rotations[4*indexs[K*idx+i] + 2], rotations[4*indexs[K*idx+i] + 3]};

        float r = rotation.x;
        float x = rotation.y;
        float y = rotation.z;
        float z = rotation.w;
        float3 normal = {2 * (x*z + r*y), 2 * (y*z - r*x), 1 - 2 * (x*x + y*y)};

        float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
        if (cos_theta < 1 && cos_theta > 1-THETA_THRESHOLD)   // 0-10度
        {
            mean_ds[indexs[K*idx]] += xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            out_loss_normal[indexs[K*idx]] += 1 - cos_theta;
        }

//         if(i == 1 && idx == 0)
//         {
//         printf("xyz[0] %f, %f, %f\n", xyz_current.x, xyz_current.y, xyz_current.z);
//         printf("xyz[1] %f, %f, %f\n", xyz.x, xyz.y, xyz.z);
//         printf("rotation[0] %f, %f, %f, %f\n", rotation_current.x, rotation_current.y, rotation_current.z, rotation_current.w);
//         printf("rotation[1] %f, %f, %f, %f\n", rotation.x, rotation.y, rotation.z, rotation.w);
//         }
    }

    mean_ds[indexs[K*idx]] /= K;

    for (int i = 0; i < K; i++)
    {
        float3 xyz = {xyzs[3*indexs[K*idx+i]], xyzs[3*indexs[K*idx+i] + 1], xyzs[3*indexs[K*idx+i] + 2]};
        float4 rotation = {rotations[4*indexs[K*idx+i]], rotations[4*indexs[K*idx+i] + 1], rotations[4*indexs[K*idx+i] + 2], rotations[4*indexs[K*idx+i] + 3]};

        float r = rotation.x;
        float x = rotation.y;
        float y = rotation.z;
        float z = rotation.w;
        float3 normal = {2 * (x*z + r*y), 2 * (y*z - r*x), 1 - 2 * (x*x + y*y)};

        float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
        if (cos_theta < 1 && cos_theta > 1-THETA_THRESHOLD)   // 0-10度
        {
            float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            out_loss_d[indexs[K*idx]] += (d - mean_ds[indexs[K*idx]]) * (d - mean_ds[indexs[K*idx]]);
        }

    }

}


__global__ void processBackwardCUDA(
        int P,
        const int* point_list_keys,
        const uint32_t* point_list,
        const float* xyzs,
        const int* xyz_ids,
        const float* rotations,
        const float* grad_out_loss_d,
        const float* grad_out_loss_normal,
        float* dL_dxyzs,
        float* dL_drotations)
{
   	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    int currentId = point_list_keys[idx];
    int nextId = (idx == P-1) ? -1 : point_list_keys[idx+1];

    if (currentId == -1 || currentId != nextId)
        return;

    float3 xyz_current = { xyzs[3*point_list[idx]], xyzs[3*point_list[idx] + 1], xyzs[3*point_list[idx] + 2] };
    float3 xyz_next = { xyzs[3*point_list[idx+1]], xyzs[3*point_list[idx+1] + 1], xyzs[3*point_list[idx+1] + 2] };
    float4 rotation_current = { rotations[4*point_list[idx]], rotations[4*point_list[idx] + 1], rotations[4*point_list[idx] + 2], rotations[4*point_list[idx] + 3] };
    float4 rotation_next = { rotations[4*point_list[idx+1]], rotations[4*point_list[idx+1] + 1], rotations[4*point_list[idx+1] + 2], rotations[4*point_list[idx+1] + 3] };

    glm::mat3 R_current = buildRotation(rotation_current);
    glm::mat3 R_next = buildRotation(rotation_next);

    float3 normal_current = {R_current[0][2], R_current[1][2], R_current[2][2]};
    float3 normal_next = {R_current[0][2], R_current[1][2], R_current[2][2]};

//     if (idx < 10)
//     {
//         printf("key id %d, %d\n", currentId, nextId);
//         printf("xyz id %d, %d\n", point_list[idx], point_list[idx+1]);
//         printf("xyz current %f, %f, %f\n", xyz_current.x, xyz_current.y, xyz_current.z);
//         printf("xyz next %f, %f, %f\n", xyz_next.x, xyz_next.y, xyz_next.z);
//         printf("rotation current %f, %f, %f, %f\n", rotation_current.x, rotation_current.y, rotation_current.z, rotation_current.w);
//         printf("rotation next %f, %f, %f, %f\n", rotation_next.x, rotation_next.y, rotation_next.z, rotation_next.w);
//     }

    float dL_dout_loss_d = grad_out_loss_d[point_list[idx]];
    float dL_dout_loss_normal = grad_out_loss_normal[point_list[idx]];

//     if (idx < 10)
//     {
//         printf("dL_dout_loss_d %f\n", dL_dout_loss_d);
//         printf("dL_dout_loss_normal %f\n", dL_dout_loss_normal);
//     }

    // depth_align 
    float dL_dd_current = dL_dout_loss_d;
    float dL_dd_next = dL_dout_loss_d;

    dL_dxyzs[3*point_list[idx]] = dL_dd_current * R_current[0][2];          //dL_dx_current
    dL_dxyzs[3*point_list[idx] + 1] = dL_dd_current * R_current[1][2];      //dL_dy_current
    dL_dxyzs[3*point_list[idx] + 2] = dL_dd_current * R_current[2][2];      //dL_dz_current
    dL_dxyzs[3*point_list[idx+1]] = dL_dd_next * R_next[0][2];          //dL_dx_next
    dL_dxyzs[3*point_list[idx+1] + 1] = dL_dd_next * R_next[1][2];      //dL_dy_next
    dL_dxyzs[3*point_list[idx+1] + 2] = dL_dd_next * R_next[2][2];      //dL_dz_next


    // depth_align + normal_align
    float cos_theta = normal_current.x*normal_next.x + normal_current.y*normal_next.y + normal_current.z*normal_next.z;
    if (cos_theta >= 1)
        return;

    float dL_dnx_current = dL_dd_current * xyz_current.x - R_next[0][2];
    float dL_dny_current = dL_dd_current * xyz_current.y - R_next[1][2];
    float dL_dnz_current = dL_dd_current * xyz_current.z - R_next[2][2];
    float dL_dnx_next = dL_dd_next * xyz_next.x - R_current[0][2];
    float dL_dny_next = dL_dd_next * xyz_next.y - R_current[1][2];
    float dL_dnz_next = dL_dd_next * xyz_next.z - R_current[2][2];

    float4 q_current = {rotation_current.y, rotation_current.z, rotation_current.w, rotation_current.x};
    float4 q_next = {rotation_next.y, rotation_next.z, rotation_next.w, rotation_next.x};
    dL_drotations[4*point_list[idx]] = 2 * q_current.y * dL_dnx_current - 2 * q_current.x * dL_dny_current;          //dL_drotation_w_current
    dL_drotations[4*point_list[idx] + 1] = 2 * q_current.z * dL_dnx_current - 2 * q_current.w * dL_dny_current - 4 * q_current.x * dL_dnz_current;      //dL_drotation_x_current
    dL_drotations[4*point_list[idx] + 2] = 2 * q_current.w * dL_dnx_current + 2 * q_current.z * dL_dny_current - 4 * q_current.y * dL_dnz_current;      //dL_drotation_y_current
    dL_drotations[4*point_list[idx] + 3] = 2 * q_current.x * dL_dnx_current + 2 * q_current.y * dL_dny_current;      //dL_drotation_z_current
    dL_drotations[4*point_list[idx+1]] = 2 * q_next.y * dL_dnx_next - 2 * q_next.x * dL_dny_next;          //dL_drotation_w_current
    dL_drotations[4*point_list[idx+1] + 1] = 2 * q_next.z * dL_dnx_next - 2 * q_next.w * dL_dny_next - 4 * q_next.x * dL_dnz_next;      //dL_drotation_x_current
    dL_drotations[4*point_list[idx+1] + 2] = 2 * q_next.w * dL_dnx_next + 2 * q_next.z * dL_dny_next - 4 * q_next.y * dL_dnz_next;      //dL_drotation_y_current
    dL_drotations[4*point_list[idx+1] + 3] = 2 * q_next.x * dL_dnx_next + 2 * q_next.y * dL_dny_next;      //dL_drotation_z_current

}

__global__ void processKNNBackwardCUDA(
        int P,
        int K,
        const float* xyzs,
        const int* xyz_ids,
        const float* rotations,
        const int* indexs,
        const float* mean_ds,
        const float* grad_out_loss_d,
        const float* grad_out_loss_normal,
        float* dL_dxyzs,
        float* dL_drotations)
{
   	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;


    float4 rotation_current = {rotations[4*indexs[K*idx]], rotations[4*indexs[K*idx] + 1], rotations[4*indexs[K*idx] + 2], rotations[4*indexs[K*idx] + 3]};
    float q_r = rotation_current.x;
    float q_x = rotation_current.y;
    float q_y = rotation_current.z;
    float q_z = rotation_current.w;
    float3 normal_current = {2 * (q_x*q_z + q_r*q_y), 2 * (q_y*q_z - q_r*q_x), 1 - 2 * (q_x*q_x + q_y*q_y)};

    float dL_dout_loss_normal = grad_out_loss_normal[indexs[K*idx]];

    for (int i = 0; i < K; i++)
    {
        float3 xyz = {xyzs[3*indexs[K*idx+i]], xyzs[3*indexs[K*idx+i] + 1], xyzs[3*indexs[K*idx+i] + 2]};
        float4 rotation = {rotations[4*indexs[K*idx+i]], rotations[4*indexs[K*idx+i] + 1], rotations[4*indexs[K*idx+i] + 2], rotations[4*indexs[K*idx+i] + 3]};

        float r = rotation.x;
        float x = rotation.y;
        float y = rotation.z;
        float z = rotation.w;
        float3 normal = {2 * (x*z + r*y), 2 * (y*z - r*x), 1 - 2 * (x*x + y*y)};


        float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
        if(cos_theta < 1 && cos_theta > 1-THETA_THRESHOLD)
        {
            // depth_align 
            float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            float dL_dout_loss_d = 2 * (d - mean_ds[indexs[K*idx]]) * grad_out_loss_d[indexs[K*idx]];
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i]]), dL_dout_loss_d * normal.x);          //dL_dx_idx_i
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i] + 1]), dL_dout_loss_d * normal.y);      //dL_dy_idx_i
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i] + 2]), dL_dout_loss_d * normal.z);      //dL_dz_idx_i

            // depth_align + normal_align
            float dL_dnx = dL_dout_loss_d * xyz.x - normal_current.x;
            float dL_dny = dL_dout_loss_d * xyz.y - normal_current.y;
            float dL_dnz = dL_dout_loss_d * xyz.z - normal_current.z;

            atomicAdd(&(dL_drotations[4*indexs[K*idx+i]]), 2 * y * dL_dnx - 2 * x * dL_dny);          //dL_drotation_w_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 1]), 2 * z * dL_dnx - 2 * r * dL_dny - 4 * x * dL_dnz);      //dL_drotation_x_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 2]), 2 * r * dL_dnx + 2 * z * dL_dny - 4 * y * dL_dnz);      //dL_drotation_y_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 3]), 2 * x * dL_dnx + 2 * y * dL_dny);      //dL_drotation_z_current
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
surfaceAlignCUDA(
	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& rotations,
    const torch::Tensor& knn_index)
{
    const int P = xyzs.size(0);
    const int k_P = knn_index.size(0);
    const int k = knn_index.size(1);
//     printf("P %d\n", P);
//     printf("k %d\n", k);
    auto int_opts = xyzs.options().dtype(torch::kInt32);
    auto float_opts = xyzs.options().dtype(torch::kFloat32);

    torch::Tensor mean_d = torch::full({P}, 0.0, float_opts);
    torch::Tensor out_loss_d = torch::full({P}, 0.0, float_opts);
    torch::Tensor out_loss_normal = torch::full({P}, 0.0, float_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));

    processKNNCUDA<<<(k_P + 255) / 256, 256>>>(k_P, k,
                                        xyzs.contiguous().data<float>(),
                                        xyz_ids.contiguous().data<int>(),
                                        rotations.contiguous().data<float>(),
                                        knn_index.contiguous().data<int>(),
                                        mean_d.contiguous().data<float>(),
                                        out_loss_d.contiguous().data<float>(),
                                        out_loss_normal.contiguous().data<float>());
    cudaDeviceSynchronize();
    return std::make_tuple(out_loss_d, out_loss_normal, binningBuffer, mean_d);
}


std::tuple<torch::Tensor, torch::Tensor>
 surfaceAlignBackwardCUDA(
 	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& rotations,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& mean_d,
    const torch::Tensor& knn_index,
 	const torch::Tensor& grad_out_loss_d,
	const torch::Tensor& grad_out_loss_normal)
{
    const int P = xyzs.size(0);
    const int k_P = knn_index.size(0);
    const int k = knn_index.size(1);

    auto int_opts = xyzs.options().dtype(torch::kInt32);
    auto float_opts = xyzs.options().dtype(torch::kFloat32);

    torch::Tensor dL_dxyzs = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor dL_drotations = torch::full({P, 4}, 0.0, float_opts);



    processKNNBackwardCUDA<<<(k_P + 255) / 256, 256>>>(k_P, k,
                                                    xyzs.contiguous().data<float>(),
                                                    xyz_ids.contiguous().data<int>(),
                                                    rotations.contiguous().data<float>(),
                                                    knn_index.contiguous().data<int>(),
                                                    mean_d.contiguous().data<float>(),
                                                    grad_out_loss_d.contiguous().data<float>(),
                                                    grad_out_loss_normal.contiguous().data<float>(),
                                                    dL_dxyzs.contiguous().data<float>(),
                                                    dL_drotations.contiguous().data<float>());
    cudaDeviceSynchronize();

    return std::make_tuple(dL_dxyzs, dL_drotations);
}