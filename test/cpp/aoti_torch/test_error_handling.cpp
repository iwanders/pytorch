
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

static void handler_that_rethrows(TorchExceptionHandle error_handle) {
  throw;
}

static std::string exception_message;
static std::string exception_message_with_backtrace;

static void handler_extract_message(TorchExceptionHandle error_handle) {
  const char* error_message = nullptr;
  EXPECT_EQ(
      aoti_torch_exception_get_what(error_handle, &error_message),
      AOTI_TORCH_SUCCESS);
  exception_message = error_message;
  error_message = nullptr;
  EXPECT_EQ(
      aoti_torch_exception_get_what_with_backtrace(
          error_handle, &error_message),
      AOTI_TORCH_SUCCESS);
  exception_message_with_backtrace = error_message;
}

TEST(AotiTorchErrorHandling, TestError) {
  // Verify that we start with the current callback equal to the default
  // callback.
  EXPECT_EQ(
      aoti_torch_exception_get_callback(),
      aoti_torch_exception_get_default_callback());

  std::cerr << "Stable header exception throws runtime_error" << std::endl;
  EXPECT_THROW(error_through_stable_header(), std::runtime_error);

  std::cerr << "Error directly with default handler " << std::endl;
  error_directly();
  std::cerr << "Should be silent after this" << std::endl;

  // Call with a nullptr, should silence the printing even.
  aoti_torch_exception_set_callback(nullptr);
  error_directly();

  // This rethrows the internal c10::Error, which subclasses from
  // std::exception, this may be helpful when you are using a debugger to get
  // additional information, it is NOT considered part of the stable ABI.
  aoti_torch_exception_set_callback(handler_that_rethrows);
  EXPECT_EQ(aoti_torch_exception_get_callback(), handler_that_rethrows);
  EXPECT_THROW(error_directly(), std::exception);

  // Verify that the messages are populated.
  aoti_torch_exception_set_callback(handler_extract_message);
  EXPECT_EQ(aoti_torch_exception_get_callback(), handler_extract_message);
  error_directly();
  EXPECT_EQ(
      exception_message,
      "The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 0");
  EXPECT_LE(exception_message.size(), exception_message_with_backtrace.size());

  std::cerr << "Restoring the exception handler." << std::endl;
  // Restore the normal exception handler.
  aoti_torch_exception_set_callback(
      aoti_torch_exception_get_default_callback());
}
