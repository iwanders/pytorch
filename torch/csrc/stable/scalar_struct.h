#pragma once


#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>



HIDDEN_NAMESPACE_BEGIN(torch, stable)


using ScalarType = torch::headeronly::ScalarType;

// The torch::stable::Scalar class is the typed tagged equivalent of at::Scalar.

/**
 * @brief A type-tagged scalar value.
 *
 * This is a type tagged wrapper necessary to put 'Scalar' arguments on the stack
 * for dispatching kernels.
 *
 * Minimum compatible version: PyTorch 2.12.
 */
class Scalar {
 private:
  /// The tag that specifies the type the value holds.
  ScalarType type_;
  /// The value that represents the scalar.
  StableIValue value_;

 public:
  Scalar(ScalarType type, StableIValue value)
      : type_(type), value_(value) {
  }

  #define DEFINE_CONSTRUCTOR(cpp_type, _unused_scalartype) \
    explicit Scalar(cpp_type value);
  AT_FORALL_SCALAR_TYPES(DEFINE_CONSTRUCTOR)


  // Copy and move constructors can be default
  /// \private
  Scalar(const Scalar& other) = default;
  /// \private
  Scalar(Scalar&& other) noexcept = default;

  // Copy and move assignment operators can be default
  /// \private
  Scalar& operator=(const Scalar& other) = default;
  /// \private
  Scalar& operator=(Scalar&& other) noexcept = default;

  // Destructor can be default
  /// \private
  ~Scalar() = default;

  /**
   * @brief Checks if two scalars are equal.
   *
   * @param other The scalar to compare with.
   * @return true if both type and values match, false otherwise.
   *
   * Minimum compatible version: PyTorch 2.12.
   */
  bool operator==(const Scalar& other) const noexcept {
    return type() == other.type() && value() == other.value();
  }

  /**
   * @brief Checks if two scalars are not equal.
   *
   * @param other The device to compare with.
   * @return true if type or value differ, false otherwise.
   *
   * Minimum compatible version: PyTorch 2.12.
   */
  bool operator!=(const Scalar& other) const noexcept {
    return !(*this == other);
  }

  /**
   * @brief Sets the value.
   *
   * @param value The new value.
   *
   * Minimum compatible version: PyTorch 2.12.
   */
  void set_value(StableIValue value) {
    value_ = value;
  }

  /**
   * @brief Returns the scalar type.
   *
   * @return The ScalarType of this device.
   *
   * Minimum compatible version: PyTorch 2.12.
   */
  ScalarType type() const noexcept {
    return type_;
  }

  /**
   * @brief Returns the value as a StableIValue
   *
   * @return The currently held value.
   *
   * Minimum compatible version: PyTorch 2.12.
   */
  StableIValue value() const noexcept {
    return value_;
  }
};

HIDDEN_NAMESPACE_END(torch, stable)
