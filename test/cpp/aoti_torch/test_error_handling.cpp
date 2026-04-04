// Checks the behaviour of:
//  aoti_torch_new_stable_ivalue
//  aoti_torch_delete_stable_ivalue

#include <gtest/gtest.h>
#include "torch/csrc/inductor/aoti_torch/c/macros.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <limits>



#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/tensor_inl.h>
#include <iostream>


using ScalarType = torch::headeronly::ScalarType;
auto dtype = torch::headeronly::ScalarType::Float;
auto device_type = torch::headeronly::DeviceType::CUDA;
auto memory_format = torch::headeronly::MemoryFormat::Contiguous;
auto layout = torch::headeronly::Layout::Strided;
using Scalar = torch::stable::Scalar;


inline torch::stable::Tensor addition(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other,
    double alpha = -1.0) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(other),
      0
    //  torch::stable::detail::from(Scalar(static_cast<int>(alpha)))
  };
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::add", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}


TEST(AotiTorchErrorHandling, TestError) {


    torch::stable::Tensor a = torch::stable::empty( {1} , ScalarType::Float);
    torch::stable::fill_(a,3.0);

    torch::stable::Tensor b = torch::stable::empty( {1}, ScalarType::Float);
    torch::stable::fill_(b,5.0);

    //torch::stable::Tensor c = torch::stable::add(a, b, 5);
    torch::stable::Tensor c = addition(a, b );

    std::cout << "a: " << *reinterpret_cast<const float*>(a.const_data_ptr()) << std::endl;
    std::cout << "b: " << *reinterpret_cast<const float*>(b.const_data_ptr()) << std::endl;
    std::cout << "c: " << *reinterpret_cast<const float*>(c.const_data_ptr()) << std::endl;
}
