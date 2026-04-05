
#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_inl.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/ScalarType.h>
#include <iostream>
#include <stdexcept>
#include "gtest/gtest.h"

using ScalarType = torch::headeronly::ScalarType;

static void error_through_stable_header() {
  torch::stable::Tensor a = torch::stable::empty({5}, ScalarType::UInt16);
  torch::stable::fill_(a, 3.0);
  torch::stable::Tensor b = torch::stable::empty({2}, ScalarType::Float);
  torch::stable::fill_(b, 5.0);
  torch::stable::Tensor c = torch::stable::subtract(a, b, 5);
}

static void error_directly() {
  torch::stable::Tensor a = torch::stable::empty({5}, ScalarType::UInt16);
  torch::stable::fill_(a, 3.0);
  torch::stable::Tensor b = torch::stable::empty({2}, ScalarType::Float);
  torch::stable::fill_(b, 5.0);
  double alpha = 3;
  AtenTensorHandle ret0;
  aoti_torch_aten_subtract_Tensor(a.get(), b.get(), alpha, &ret0);
}

TEST(AotiTorchErrorHandling, TestError) {


  std::cerr << "Stable header exception throws runtime_error" << std::endl;
  EXPECT_THROW(error_through_stable_header(), std::runtime_error);

  try {
    error_through_stable_header();
  } catch (const std::runtime_error& e){
    EXPECT_EQ(std::string(e.what()), "1 API call failed at /workspace/pytorch/torch/csrc/stable/ops.h, line 1058, with: The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 0");

  }

  std::cerr << "Error directly with default handler " << std::endl;
  error_directly();

  EXPECT_EQ(std::string(aoti_torch_exception_get_what()),"The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 0");

  std::cout << aoti_torch_exception_get_what() << std::endl;
  std::cout << aoti_torch_exception_get_what_with_backtrace() << std::endl;
}
