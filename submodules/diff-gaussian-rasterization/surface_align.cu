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

#define THETA_THRESHOLD 0.004       // cosine of 5 degrees

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
        if (cos_theta < 1 && cos_theta > 1-THETA_THRESHOLD)   // 0-5 degrees
        {
            mean_ds[indexs[K*idx]] += xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            out_loss_normal[indexs[K*idx]] += 1 - cos_theta;
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
        if (cos_theta < 1 && cos_theta > 1-THETA_THRESHOLD)   // 0-5 degrees
        {
            float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            out_loss_d[indexs[K*idx]] += (d - mean_ds[indexs[K*idx]]) * (d - mean_ds[indexs[K*idx]]);
        }

    }

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
            // Depth alignment
            float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            float dL_dout_loss_d = 2 * (d - mean_ds[indexs[K*idx]]) * grad_out_loss_d[indexs[K*idx]];
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i]]), dL_dout_loss_d * normal.x);          //dL_dx_idx_i
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i] + 1]), dL_dout_loss_d * normal.y);      //dL_dy_idx_i
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i] + 2]), dL_dout_loss_d * normal.z);      //dL_dz_idx_i

            // Depth alignment + Normal alignment
            float dL_dnx = dL_dout_loss_d * xyz.x - normal_current.x;
            float dL_dny = dL_dout_loss_d * xyz.y - normal_current.y;
            float dL_dnz = dL_dout_loss_d * xyz.z - normal_current.z;

            atomicAdd(&(dL_drotations[4*indexs[K*idx+i]]), 2 * y * dL_dnx - 2 * x * dL_dny);                        //dL_drotation_w_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 1]), 2 * z * dL_dnx - 2 * r * dL_dny - 4 * x * dL_dnz);   //dL_drotation_x_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 2]), 2 * r * dL_dnx + 2 * z * dL_dny - 4 * y * dL_dnz);   //dL_drotation_y_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 3]), 2 * x * dL_dnx + 2 * y * dL_dny);                    //dL_drotation_z_current
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