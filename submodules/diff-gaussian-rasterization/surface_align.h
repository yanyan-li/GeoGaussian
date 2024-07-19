#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

template <typename T>
static void obtain_(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
	std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
	ptr = reinterpret_cast<T*>(offset);
	chunk = reinterpret_cast<char*>(ptr + count);
}

struct BinningState
{
	size_t sorting_size;
	int* point_list_keys_unsorted;
	int* point_list_keys;
	uint32_t* point_list_unsorted;
	uint32_t* point_list;
	char* list_sorting_space;

	static BinningState fromChunk(char*& chunk, size_t P);
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
surfaceAlignCUDA(
	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& rotations,
    const torch::Tensor& knn_index);

std::tuple<torch::Tensor, torch::Tensor>
 surfaceAlignBackwardCUDA(
 	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& rotations,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& mean_d,
    const torch::Tensor& knn_index,
 	const torch::Tensor& grad_out_loss_d,
	const torch::Tensor& grad_out_loss_normal);