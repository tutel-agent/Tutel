// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace tutel {
namespace fused_moe_nvfp4 {

constexpr std::array<float, 16> kNvfp4E2m1 = {{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
}};

alignas(64) constexpr std::array<uint16_t, 32> kNvfp4E2m1Bf16 = {{
    0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
    0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0,
    0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
    0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0,
}};

inline float nvfp4_e4m3fn_to_float(uint8_t bits) {
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;
  float value;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -9);
  } else if (exponent == 0x0f && mantissa == 0x07) {
    value = std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
  }
  return (bits & 0x80) ? -value : value;
}

inline const std::array<float, 256>& nvfp4_e4m3fn_lut() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> result{};
    for (int i = 0; i < 256; ++i) {
      result[i] = nvfp4_e4m3fn_to_float(static_cast<uint8_t>(i));
    }
    return result;
  }();
  return table;
}

inline float nvfp4_dot_scalar(
    const float* activation_even,
    const float* activation_odd,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    int64_t packed_k) {
  const auto& scale_lut = nvfp4_e4m3fn_lut();
  float sum = 0.0f;
  for (int64_t p = 0; p < packed_k; ++p) {
    const uint8_t packed = packed_weight[p];
    const float block_scale = scale_lut[scale[p / 8]];
    sum += activation_even[p] * (kNvfp4E2m1[packed & 0x0f] * block_scale) +
        activation_odd[p] * (kNvfp4E2m1[packed >> 4] * block_scale);
  }
  return sum;
}

inline uint16_t nvfp4_bf16_magnitude_limit(float safe_activation) {
  uint32_t bits;
  std::memcpy(&bits, &safe_activation, sizeof(bits));
  // Truncate, not round: the largest allowed BF16 must not exceed the FP32 bound.
  return static_cast<uint16_t>(bits >> 16);
}

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#define TUTEL_NVFP4_AVX2_AVAILABLE 1
#define TUTEL_NVFP4_AVX2_TARGET __attribute__((target("avx2,fma")))
#elif (defined(_M_X64) || defined(_M_IX86)) && \
    (defined(__AVX2__) || defined(_M_AVX2))
#define TUTEL_NVFP4_AVX2_AVAILABLE 1
#define TUTEL_NVFP4_AVX2_TARGET
#else
#define TUTEL_NVFP4_AVX2_AVAILABLE 0
#define TUTEL_NVFP4_AVX2_TARGET
#endif

#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__) && \
    ((defined(__clang__) && __clang_major__ >= 12) || \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11))
#define TUTEL_NVFP4_AVX512_BF16_AVAILABLE 1
#define TUTEL_NVFP4_AVX512_BF16_TARGET \
  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16,fma")))
#else
#define TUTEL_NVFP4_AVX512_BF16_AVAILABLE 0
#define TUTEL_NVFP4_AVX512_BF16_TARGET
#endif

#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
inline bool nvfp4_cpu_supports_avx512_bf16() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") &&
      __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512vl") &&
      __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512bf16") &&
      __builtin_cpu_supports("fma");
}

TUTEL_NVFP4_AVX512_BF16_TARGET inline __m512 nvfp4_accumulate_avx512_bf16(
    __m512 sum,
    __m512i activation_bits,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    __m256i nibble_mask,
    __m512i e2m1_lut,
    const float* scale_lut) {
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_weight));
  const __m256i packed_16 = _mm256_cvtepu8_epi16(packed);
  const __m256i even_indices = _mm256_and_si256(packed_16, nibble_mask);
  const __m256i odd_indices = _mm256_srli_epi16(packed_16, 4);
  __m512i indices = _mm512_castsi256_si512(even_indices);
  indices = _mm512_inserti64x4(indices, odd_indices, 1);
  const __m512i weight_bits = _mm512_permutexvar_epi16(indices, e2m1_lut);
  const __m512 contribution = _mm512_dpbf16_ps(
      _mm512_setzero_ps(), (__m512bh)activation_bits, (__m512bh)weight_bits);
  const __m512 scale0 = _mm512_set1_ps(scale_lut[scale[0]]);
  const __m512 scale1 = _mm512_set1_ps(scale_lut[scale[1]]);
  const __m512 scale_pattern = _mm512_mask_blend_ps(0xf0f0, scale0, scale1);
  return _mm512_fmadd_ps(contribution, scale_pattern, sum);
}

TUTEL_NVFP4_AVX512_BF16_TARGET inline float nvfp4_dot_avx512_bf16(
    const c10::BFloat16* activation_permuted,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    int64_t packed_k) {
  const __m256i nibble_mask = _mm256_set1_epi16(0x0f);
  const __m512i e2m1_lut = _mm512_load_si512(kNvfp4E2m1Bf16.data());
  const auto& scale_lut = nvfp4_e4m3fn_lut();
  __m512 sum = _mm512_setzero_ps();

  for (int64_t p = 0; p < packed_k; p += 16) {
    const __m512i activation_bits = _mm512_loadu_si512(
        reinterpret_cast<const void*>(activation_permuted + 2 * p));
    sum = nvfp4_accumulate_avx512_bf16(
        sum, activation_bits, packed_weight + p, scale + p / 8,
        nibble_mask, e2m1_lut, scale_lut.data());
  }
  return _mm512_reduce_add_ps(sum);
}

TUTEL_NVFP4_AVX512_BF16_TARGET inline void nvfp4_dot_rows_avx512_bf16(
    const c10::BFloat16* activation_permuted,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    int64_t packed_k,
    int64_t row_count,
    float* output) {
  const __m256i nibble_mask = _mm256_set1_epi16(0x0f);
  const __m512i e2m1_lut = _mm512_load_si512(kNvfp4E2m1Bf16.data());
  const float* scale_lut = nvfp4_e4m3fn_lut().data();
  const int64_t scale_k = packed_k / 8;
  int64_t row = 0;
  for (; row + 4 <= row_count; row += 4) {
    const uint8_t* weight0 = packed_weight + row * packed_k;
    const uint8_t* scale0 = scale + row * scale_k;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    // Interleave independent rows without reassociating any row's K sum.
    for (int64_t p = 0; p < packed_k; p += 16) {
      const __m512i activation_bits = _mm512_loadu_si512(
          reinterpret_cast<const void*>(activation_permuted + 2 * p));
      sum0 = nvfp4_accumulate_avx512_bf16(
          sum0, activation_bits, weight0 + p, scale0 + p / 8,
          nibble_mask, e2m1_lut, scale_lut);
      sum1 = nvfp4_accumulate_avx512_bf16(
          sum1, activation_bits, weight0 + packed_k + p, scale0 + scale_k + p / 8,
          nibble_mask, e2m1_lut, scale_lut);
      sum2 = nvfp4_accumulate_avx512_bf16(
          sum2, activation_bits, weight0 + 2 * packed_k + p,
          scale0 + 2 * scale_k + p / 8, nibble_mask, e2m1_lut, scale_lut);
      sum3 = nvfp4_accumulate_avx512_bf16(
          sum3, activation_bits, weight0 + 3 * packed_k + p,
          scale0 + 3 * scale_k + p / 8, nibble_mask, e2m1_lut, scale_lut);
    }
    output[row] = _mm512_reduce_add_ps(sum0);
    output[row + 1] = _mm512_reduce_add_ps(sum1);
    output[row + 2] = _mm512_reduce_add_ps(sum2);
    output[row + 3] = _mm512_reduce_add_ps(sum3);
  }
  for (; row < row_count; ++row) {
    output[row] = nvfp4_dot_avx512_bf16(
        activation_permuted, packed_weight + row * packed_k,
        scale + row * scale_k, packed_k);
  }
}
#endif

#if TUTEL_NVFP4_AVX2_AVAILABLE
inline bool nvfp4_cpu_supports_avx2_fma() {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return true;
#endif
}

TUTEL_NVFP4_AVX2_TARGET inline bool nvfp4_prepare_bf16_avx2(
    const c10::BFloat16* activation,
    c10::BFloat16* permuted,
    int64_t count,
    uint16_t magnitude_limit) {
  const __m256i sign_mask = _mm256_set1_epi16(0x7fff);
  const __m256i shuffle = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
  __m256i max_magnitude = _mm256_setzero_si256();
  // K is a multiple of 32. Each block becomes the even/odd halves consumed
  // by the AVX512 dot kernel; unsigned BF16 magnitudes also reject Inf/NaN.
  for (int64_t index = 0; index < count; index += 32) {
    const __m256i first = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activation + index));
    const __m256i second = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activation + index + 16));
    max_magnitude = _mm256_max_epu16(
        max_magnitude,
        _mm256_max_epu16(
            _mm256_and_si256(first, sign_mask),
            _mm256_and_si256(second, sign_mask)));
    const __m256i first_halves = _mm256_permute4x64_epi64(
        _mm256_shuffle_epi8(first, shuffle), _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i second_halves = _mm256_permute4x64_epi64(
        _mm256_shuffle_epi8(second, shuffle), _MM_SHUFFLE(3, 1, 2, 0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(permuted + index),
        _mm256_permute2x128_si256(first_halves, second_halves, 0x20));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(permuted + index + 16),
        _mm256_permute2x128_si256(first_halves, second_halves, 0x31));
  }
  const __m256i limit = _mm256_set1_epi16(static_cast<short>(magnitude_limit));
  return _mm256_movemask_epi8(_mm256_cmpeq_epi16(
      _mm256_max_epu16(max_magnitude, limit), limit)) == -1;
}

TUTEL_NVFP4_AVX2_TARGET inline float nvfp4_dot_avx2(
    const float* activation_even,
    const float* activation_odd,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    int64_t packed_k) {
  const __m128i nibble_mask = _mm_set1_epi8(0x0f);
  const __m128i e2m1_lut = _mm_setr_epi8(
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  const auto& scale_lut = nvfp4_e4m3fn_lut();

  int64_t p = 0;
  for (; p + 16 <= packed_k; p += 16) {
    const __m128i packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_weight + p));
    const __m128i low = _mm_shuffle_epi8(e2m1_lut, _mm_and_si128(packed, nibble_mask));
    const __m128i high = _mm_shuffle_epi8(
        e2m1_lut, _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask));

    const __m256 low0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low));
    const __m256 low1 =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low, 8)));
    const __m256 high0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high));
    const __m256 high1 =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high, 8)));

    const __m256 scale0 = _mm256_set1_ps(scale_lut[scale[p / 8]] * 0.5f);
    const __m256 scale1 = _mm256_set1_ps(scale_lut[scale[p / 8 + 1]] * 0.5f);

    sum0 = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_even + p), _mm256_mul_ps(low0, scale0), sum0);
    sum0 = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_odd + p), _mm256_mul_ps(high0, scale0), sum0);
    sum1 = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_even + p + 8), _mm256_mul_ps(low1, scale1), sum1);
    sum1 = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_odd + p + 8), _mm256_mul_ps(high1, scale1), sum1);
  }

  const __m256 sum = _mm256_add_ps(sum0, sum1);
  const __m128 halves =
      _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
  const __m128 pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
  float result = _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
  _mm256_zeroupper();

  for (; p < packed_k; ++p) {
    const uint8_t packed = packed_weight[p];
    const float block_scale = scale_lut[scale[p / 8]];
    result += activation_even[p] * (kNvfp4E2m1[packed & 0x0f] * block_scale) +
        activation_odd[p] * (kNvfp4E2m1[packed >> 4] * block_scale);
  }
  return result;
}
#endif

inline float nvfp4_silu(float value) {
  if (value >= 0.0f) {
    return value / (1.0f + std::exp(-value));
  }
  const float exp_value = std::exp(value);
  return value * exp_value / (1.0f + exp_value);
}

inline void nvfp4_validate_output_scale(double output_scale, const char* name) {
  TORCH_CHECK(
      std::isfinite(output_scale),
      name, " must be finite, got ", output_scale);
  TORCH_CHECK(
      std::abs(output_scale) <= std::numeric_limits<float>::max(),
      name, " must be representable as a finite float, got ", output_scale);
}

struct Nvfp4PreparedActivations {
  int64_t rows;
  int64_t K;
  int64_t packed_k;
  int64_t row_tile;
#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
  bool use_avx512_bf16;
  std::vector<c10::BFloat16> avx512_bf16;
#endif
#if TUTEL_NVFP4_AVX2_AVAILABLE
  bool use_avx2;
#endif
  std::vector<float> even;
  std::vector<float> odd;

  Nvfp4PreparedActivations(
      const c10::BFloat16* activation,
      int64_t activation_rows,
      int64_t activation_k,
      int64_t activation_row_tile = 1)
      : rows(activation_rows), K(activation_k), packed_k(activation_k / 2),
        row_tile(activation_row_tile)
#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
      , use_avx512_bf16(
          activation_k % 32 == 0 && nvfp4_cpu_supports_avx512_bf16())
#endif
#if TUTEL_NVFP4_AVX2_AVAILABLE
      , use_avx2(nvfp4_cpu_supports_avx2_fma())
#endif
  {
    // Bound the absolute sum so vector lane grouping cannot overflow where
    // scalar-order activation * (FP4 * E4M3FN) accumulation stays finite.
    const double maximum_dot_term =
        6.0 * 448.0 * static_cast<double>(K);
    const float safe_activation = static_cast<float>(
        static_cast<double>(std::numeric_limits<float>::max()) /
        maximum_dot_term);
    bool vector_values_safe = true;
#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
    if (use_avx512_bf16) {
      avx512_bf16.resize(rows * K);
      vector_values_safe = nvfp4_prepare_bf16_avx2(
          activation, avx512_bf16.data(), rows * K,
          nvfp4_bf16_magnitude_limit(safe_activation));
      if (vector_values_safe) {
        return;
      }
      use_avx512_bf16 = false;
    } else
#endif
    {
      for (int64_t index = 0; index < rows * K; ++index) {
        const float value = static_cast<float>(activation[index]);
        if (!std::isfinite(value) || std::abs(value) > safe_activation) {
          vector_values_safe = false;
          break;
        }
      }
    }
#if TUTEL_NVFP4_AVX2_AVAILABLE
    use_avx2 = use_avx2 && vector_values_safe;
#endif
    even.resize(rows * packed_k);
    odd.resize(rows * packed_k);
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t source_offset = row * K;
      const int64_t target_offset = row * packed_k;
      for (int64_t p = 0; p < packed_k; ++p) {
        even[target_offset + p] =
            static_cast<float>(activation[source_offset + 2 * p]);
        odd[target_offset + p] =
            static_cast<float>(activation[source_offset + 2 * p + 1]);
      }
    }
  }

  const char* backend_name() const {
#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
    if (use_avx512_bf16) {
      return row_tile == 4 ? "AVX512-BF16/rows4" : "AVX512-BF16";
    }
#endif
#if TUTEL_NVFP4_AVX2_AVAILABLE
    if (use_avx2) {
      return "AVX2";
    }
#endif
    return "scalar";
  }

  bool uses_row_tile() const {
#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
    return use_avx512_bf16 && row_tile == 4;
#else
    return false;
#endif
  }

  inline float dot(
      int64_t row,
      const uint8_t* weight,
      const uint8_t* scale) const {
#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
    if (use_avx512_bf16) {
      return nvfp4_dot_avx512_bf16(
          avx512_bf16.data() + row * K, weight, scale, packed_k);
    }
#endif
#if TUTEL_NVFP4_AVX2_AVAILABLE
    if (use_avx2) {
      return nvfp4_dot_avx2(
          even.data() + row * packed_k,
          odd.data() + row * packed_k,
          weight, scale, packed_k);
    }
#endif
    return nvfp4_dot_scalar(
        even.data() + row * packed_k,
        odd.data() + row * packed_k,
        weight, scale, packed_k);
  }

  inline void dot_rows(
      int64_t row,
      const uint8_t* weight,
      const uint8_t* scale,
      int64_t row_count,
      float* output) const {
#if TUTEL_NVFP4_AVX512_BF16_AVAILABLE
    if (uses_row_tile()) {
      nvfp4_dot_rows_avx512_bf16(
          avx512_bf16.data() + row * K, weight, scale, packed_k, row_count, output);
      return;
    }
#endif
    for (int64_t n = 0; n < row_count; ++n) {
      output[n] = dot(row, weight + n * packed_k, scale + n * (K / 16));
    }
  }
};

template <bool kSwiGLU, bool kSharedActivation, bool kReduce>
inline torch::Tensor nvfp4_batched_gemv_impl(
    const torch::Tensor& A_cpu,
    const torch::Tensor& W_cpu,
    const torch::Tensor& W_scale_cpu,
    const torch::Tensor& expert_ids_cpu,
    const torch::Tensor& expert_weights_cpu,
    double output_scale,
    const char* op) {
  TORCH_CHECK(A_cpu.device().is_cpu(), op, ": A_cpu must be on CPU, got ", A_cpu.device());
  TORCH_CHECK(W_cpu.device().is_cpu(), op, ": W_cpu must be on CPU, got ", W_cpu.device());
  TORCH_CHECK(
      W_scale_cpu.device().is_cpu(), op, ": W_scale_cpu must be on CPU, got ",
      W_scale_cpu.device());
  TORCH_CHECK(
      expert_ids_cpu.device().is_cpu(), op, ": expert_ids_cpu must be on CPU, got ",
      expert_ids_cpu.device());
  TORCH_CHECK(A_cpu.is_contiguous(), op, ": A_cpu must be contiguous");
  TORCH_CHECK(W_cpu.is_contiguous(), op, ": W_cpu must be contiguous");
  TORCH_CHECK(W_scale_cpu.is_contiguous(), op, ": W_scale_cpu must be contiguous");
  TORCH_CHECK(expert_ids_cpu.is_contiguous(), op, ": expert_ids_cpu must be contiguous");
  TORCH_CHECK(
      A_cpu.scalar_type() == at::kBFloat16, op, ": A_cpu must have dtype torch.bfloat16, got ",
      A_cpu.scalar_type());
  TORCH_CHECK(
      W_cpu.scalar_type() == at::kByte, op, ": W_cpu must have dtype torch.uint8, got ",
      W_cpu.scalar_type());
  TORCH_CHECK(
      W_scale_cpu.scalar_type() == at::kByte,
      op, ": W_scale_cpu must have dtype torch.uint8, got ", W_scale_cpu.scalar_type());
  TORCH_CHECK(
      expert_ids_cpu.scalar_type() == at::kInt,
      op, ": expert_ids_cpu must have dtype torch.int32, got ", expert_ids_cpu.scalar_type());
  TORCH_CHECK(
      A_cpu.dim() == 2, op, ": A_cpu must have shape [activation_rows, K], got ",
      A_cpu.sizes());
  TORCH_CHECK(
      W_cpu.dim() == 3, op, ": W_cpu must have shape [E, N, K/2], got ", W_cpu.sizes());
  TORCH_CHECK(
      W_scale_cpu.dim() == 3,
      op, ": W_scale_cpu must have shape [E, N, K/16], got ", W_scale_cpu.sizes());
  TORCH_CHECK(
      expert_ids_cpu.dim() == 1,
      op, ": expert_ids_cpu must have shape [batch], got ", expert_ids_cpu.sizes());
  const int64_t batch = expert_ids_cpu.size(0);
  if constexpr (kReduce) {
    TORCH_CHECK(
        expert_weights_cpu.device().is_cpu(),
        op, ": expert_weights_cpu must be on CPU, got ", expert_weights_cpu.device());
    TORCH_CHECK(
        expert_weights_cpu.is_contiguous(),
        op, ": expert_weights_cpu must be contiguous");
    TORCH_CHECK(
        expert_weights_cpu.scalar_type() == at::kFloat,
        op, ": expert_weights_cpu must have dtype torch.float32, got ",
        expert_weights_cpu.scalar_type());
    TORCH_CHECK(
        expert_weights_cpu.dim() == 1 && expert_weights_cpu.size(0) == batch,
        op, ": expert_weights_cpu must have shape [", batch, "], got ",
        expert_weights_cpu.sizes());
  }
  const int64_t activation_rows = kSharedActivation ? 1 : batch;
  TORCH_CHECK(
      A_cpu.size(0) == activation_rows,
      op, ": A_cpu first dimension must be ", activation_rows, ", got ", A_cpu.size(0));

  const int64_t E = W_cpu.size(0);
  const int64_t weight_rows = W_cpu.size(1);
  const int64_t K = A_cpu.size(1);
  TORCH_CHECK(E > 0, op, ": E must be positive, got ", E);
  TORCH_CHECK(weight_rows > 0, op, ": W_cpu output dimension must be positive");
  TORCH_CHECK(
      !kSwiGLU || weight_rows % 2 == 0,
      op, ": W13 output dimension must be even ([gate, up]), got ", weight_rows);
  const int64_t N = kSwiGLU ? weight_rows / 2 : weight_rows;
  TORCH_CHECK(K > 0 && K % 16 == 0, op, ": K must be positive and divisible by 16, got ", K);
  TORCH_CHECK(batch > 0, op, ": batch must be positive, got ", batch);
  TORCH_CHECK(
      W_cpu.size(2) == K / 2,
      op, ": W_cpu shape mismatch: expected [", E, ", ", weight_rows, ", ", K / 2,
      "], got ", W_cpu.sizes());
  TORCH_CHECK(
      W_scale_cpu.size(0) == E && W_scale_cpu.size(1) == weight_rows &&
          W_scale_cpu.size(2) == K / 16,
      op, ": W_scale_cpu shape mismatch: expected [", E, ", ", weight_rows, ", ", K / 16,
      "], got ", W_scale_cpu.sizes());
  TORCH_CHECK(std::isfinite(output_scale), op, ": output_scale must be finite, got ", output_scale);
  TORCH_CHECK(
      std::abs(output_scale) <= std::numeric_limits<float>::max(),
      op, ": output_scale must be representable as a finite float, got ", output_scale);

  const int32_t* expert_ids = expert_ids_cpu.data_ptr<int32_t>();
  for (int64_t b = 0; b < batch; ++b) {
    TORCH_CHECK(
        expert_ids[b] >= 0 && expert_ids[b] < E,
        op, ": expert_ids_cpu[", b, "] must be in [0, ", E, "), got ", expert_ids[b]);
  }
  const float* expert_weights = nullptr;
  if constexpr (kReduce) {
    expert_weights = expert_weights_cpu.data_ptr<float>();
    for (int64_t b = 0; b < batch; ++b) {
      TORCH_CHECK(
          std::isfinite(expert_weights[b]),
          op, ": expert_weights_cpu[", b, "] must be finite, got ", expert_weights[b]);
    }
  }

  const int64_t packed_k = K / 2;
  const Nvfp4PreparedActivations prepared(
      A_cpu.data_ptr<c10::BFloat16>(), activation_rows, K);

  const uint8_t* weights = W_cpu.data_ptr<uint8_t>();
  const uint8_t* scales = W_scale_cpu.data_ptr<uint8_t>();
  auto output = torch::empty({kReduce ? 1 : batch, N}, A_cpu.options());
  c10::BFloat16* output_data = output.data_ptr<c10::BFloat16>();
  const float output_scale_f = static_cast<float>(output_scale);

  const auto dot = [&](int64_t activation_row, const uint8_t* weight, const uint8_t* scale) {
    return prepared.dot(activation_row, weight, scale);
  };

  if constexpr (kReduce) {
    const int64_t parallel_grain =
        std::max<int64_t>(1, N / std::max(1, at::get_num_threads()));
    at::parallel_for(0, N, parallel_grain, [&](int64_t begin, int64_t end) {
      std::vector<float> accumulator(end - begin, 0.0f);
      for (int64_t b = 0; b < batch; ++b) {
        const int64_t expert_row = expert_ids[b] * weight_rows;
        const float routing_weight = expert_weights[b];
        for (int64_t n = begin; n < end; ++n) {
          const int64_t weight_row = expert_row + n;
          const float projected = dot(
              b,
              weights + weight_row * packed_k,
              scales + weight_row * (K / 16)) * output_scale_f;
          // Preserve projection-scale and routing multiplication order in FP32.
          accumulator[n - begin] += projected * routing_weight;
        }
      }
      for (int64_t n = begin; n < end; ++n) {
        output_data[n] = c10::BFloat16(accumulator[n - begin]);
      }
    });
  } else {
    const int64_t output_rows = batch * N;
    const int64_t parallel_grain =
        std::max<int64_t>(1, output_rows / std::max(1, at::get_num_threads()));
    at::parallel_for(0, output_rows, parallel_grain, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        const int64_t b = row / N;
        const int64_t n = row - b * N;
        const int64_t expert = expert_ids[b];
        const int64_t gate_row = expert * weight_rows + n;
        const uint8_t* row_weight = weights + gate_row * packed_k;
        const uint8_t* row_scale = scales + gate_row * (K / 16);
        const int64_t activation_row = kSharedActivation ? 0 : b;

        const float projected =
            dot(activation_row, row_weight, row_scale) * output_scale_f;
        if constexpr (kSwiGLU) {
          // output_scale belongs to each W13 projection and cannot move across SiLU.
          const float gate = projected;
          const int64_t up_row = gate_row + N;
          const float up = dot(
              activation_row,
              weights + up_row * packed_k,
              scales + up_row * (K / 16)) * output_scale_f;
          output_data[row] = c10::BFloat16(nvfp4_silu(gate) * up);
        } else {
          output_data[row] = c10::BFloat16(projected);
        }
      }
    });
  }
  return output;
}

inline torch::Tensor nvfp4_batched_gemv_cpu(
    const torch::Tensor& A_cpu,
    const torch::Tensor& W_cpu,
    const torch::Tensor& W_scale_cpu,
    const torch::Tensor& expert_ids_cpu,
    double output_scale) {
  return nvfp4_batched_gemv_impl<false, true, false>(
      A_cpu, W_cpu, W_scale_cpu, expert_ids_cpu, torch::Tensor(), output_scale,
      "tutel_ops::nvfp4_batched_gemv");
}

inline torch::Tensor nvfp4_batched_gemv_swiglu_cpu(
    const torch::Tensor& A_cpu,
    const torch::Tensor& W_cpu,
    const torch::Tensor& W_scale_cpu,
    const torch::Tensor& expert_ids_cpu,
    double output_scale) {
  return nvfp4_batched_gemv_impl<true, true, false>(
      A_cpu, W_cpu, W_scale_cpu, expert_ids_cpu, torch::Tensor(), output_scale,
      "tutel_ops::nvfp4_batched_gemv_swiglu");
}

inline torch::Tensor nvfp4_batched_gemv_w2_cpu(
    const torch::Tensor& A_cpu,
    const torch::Tensor& W_cpu,
    const torch::Tensor& W_scale_cpu,
    const torch::Tensor& expert_ids_cpu,
    double output_scale) {
  return nvfp4_batched_gemv_impl<false, false, false>(
      A_cpu, W_cpu, W_scale_cpu, expert_ids_cpu, torch::Tensor(), output_scale,
      "tutel_ops::nvfp4_batched_gemv_w2");
}

inline torch::Tensor nvfp4_batched_gemv_w2_reduce_cpu(
    const torch::Tensor& A_cpu,
    const torch::Tensor& W_cpu,
    const torch::Tensor& W_scale_cpu,
    const torch::Tensor& expert_ids_cpu,
    const torch::Tensor& expert_weights_cpu,
    double output_scale) {
  return nvfp4_batched_gemv_impl<false, false, true>(
      A_cpu, W_cpu, W_scale_cpu, expert_ids_cpu, expert_weights_cpu, output_scale,
      "tutel_ops::nvfp4_batched_gemv_w2_reduce");
}

inline int64_t nvfp4_fused_row_tile() {
  const char* value = std::getenv("TUTEL_NVFP4_ROW_TILE");
  if (value == nullptr || std::strcmp(value, "1") == 0) {
    return 1;
  }
  TORCH_CHECK(
      std::strcmp(value, "4") == 0,
      "TUTEL_NVFP4_ROW_TILE must be 1 or 4, got '", value, "'");
  return 4;
}

struct Nvfp4MoeProfile {
  using Clock = std::chrono::steady_clock;
  Clock::time_point last;
  std::array<double, 6> seconds{};
  const char* w13_backend = "";
  const char* w2_backend = "";

  void record(size_t phase) {
    const auto now = Clock::now();
    seconds[phase] = std::chrono::duration<double>(now - last).count();
    last = now;
  }
};

template <bool kProfile>
inline torch::Tensor fused_nvfp4_moe_swiglu_cpu_impl(
    const torch::Tensor& x,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    double w13_output_scale,
    double w2_output_scale,
    Nvfp4MoeProfile* profile) {
  if constexpr (kProfile) {
    profile->last = Nvfp4MoeProfile::Clock::now();
  }
  const int64_t row_tile = nvfp4_fused_row_tile();
  TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
  TORCH_CHECK(w13.device().is_cpu(), "w13 must be a CPU tensor");
  TORCH_CHECK(w13_scale.device().is_cpu(), "w13_scale must be a CPU tensor");
  TORCH_CHECK(w2.device().is_cpu(), "w2 must be a CPU tensor");
  TORCH_CHECK(w2_scale.device().is_cpu(), "w2_scale must be a CPU tensor");
  TORCH_CHECK(topk_ids.device().is_cpu(), "topk_ids must be a CPU tensor");
  TORCH_CHECK(
      topk_weights.device().is_cpu(), "topk_weights must be a CPU tensor");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w13.is_contiguous(), "w13 must be contiguous");
  TORCH_CHECK(w13_scale.is_contiguous(), "w13_scale must be contiguous");
  TORCH_CHECK(w2.is_contiguous(), "w2 must be contiguous");
  TORCH_CHECK(w2_scale.is_contiguous(), "w2_scale must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(
      topk_weights.is_contiguous(), "topk_weights must be contiguous");

  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must have dtype torch.bfloat16");
  TORCH_CHECK(w13.scalar_type() == at::kByte, "w13 must have dtype torch.uint8");
  TORCH_CHECK(
      w13_scale.scalar_type() == at::kByte,
      "w13_scale must have dtype torch.uint8");
  TORCH_CHECK(w2.scalar_type() == at::kByte, "w2 must have dtype torch.uint8");
  TORCH_CHECK(
      w2_scale.scalar_type() == at::kByte,
      "w2_scale must have dtype torch.uint8");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt ||
          topk_ids.scalar_type() == at::kLong,
      "topk_ids must have dtype torch.int32 or torch.int64");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "topk_weights must have dtype torch.float32");

  TORCH_CHECK(x.dim() == 2, "x must have shape [M, K], got ", x.sizes());
  TORCH_CHECK(
      w13.dim() == 3,
      "w13 must have shape [E, 2*I, K/2], got ", w13.sizes());
  TORCH_CHECK(
      w13_scale.dim() == 3,
      "w13_scale must have shape [E, 2*I, K/16], got ",
      w13_scale.sizes());
  TORCH_CHECK(
      w2.dim() == 3,
      "w2 must have shape [E, N, I/2], got ", w2.sizes());
  TORCH_CHECK(
      w2_scale.dim() == 3,
      "w2_scale must have shape [E, N, I/16], got ", w2_scale.sizes());
  TORCH_CHECK(
      topk_ids.dim() == 2,
      "topk_ids must have shape [M, T], got ", topk_ids.sizes());
  TORCH_CHECK(
      topk_weights.dim() == 2,
      "topk_weights must have shape [M, T], got ", topk_weights.sizes());

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t E = w13.size(0);
  const int64_t w13_rows = w13.size(1);
  const int64_t I = w13_rows / 2;
  const int64_t N = w2.size(1);
  const int64_t T = topk_ids.size(1);
  TORCH_CHECK(M > 0, "x dimension M must be positive");
  TORCH_CHECK(
      K > 0 && K % 16 == 0,
      "x dimension K must be positive and divisible by 16, got ", K);
  TORCH_CHECK(E > 0, "w13 expert dimension E must be positive");
  TORCH_CHECK(
      w13_rows > 0 && w13_rows % 2 == 0,
      "w13 dimension 1 must be positive and even ([gate, up]), got ",
      w13_rows);
  TORCH_CHECK(
      I % 16 == 0,
      "intermediate dimension I must be divisible by 16, got ", I);
  TORCH_CHECK(N > 0, "w2 output dimension N must be positive");
  TORCH_CHECK(T > 0, "top-k dimension T must be positive");
  TORCH_CHECK(
      T <= std::numeric_limits<int64_t>::max() / M,
      "M*T is too large");

  TORCH_CHECK(
      w13.size(2) == K / 2,
      "w13 shape mismatch: expected dimension 2 to be K/2=", K / 2,
      ", got ", w13.size(2));
  TORCH_CHECK(
      w13_scale.size(0) == E &&
          w13_scale.size(1) == w13_rows &&
          w13_scale.size(2) == K / 16,
      "w13_scale must have shape [", E, ", ", w13_rows, ", ", K / 16,
      "], got ", w13_scale.sizes());
  TORCH_CHECK(
      w2.size(0) == E,
      "w2 expert dimension must match w13 E=", E, ", got ", w2.size(0));
  TORCH_CHECK(
      w2.size(2) == I / 2,
      "w2 shape mismatch: expected dimension 2 to be I/2=", I / 2,
      ", got ", w2.size(2));
  TORCH_CHECK(
      w2_scale.size(0) == E &&
          w2_scale.size(1) == N &&
          w2_scale.size(2) == I / 16,
      "w2_scale must have shape [", E, ", ", N, ", ", I / 16,
      "], got ", w2_scale.sizes());
  TORCH_CHECK(
      topk_ids.size(0) == M,
      "topk_ids dimension 0 must match x M=", M, ", got ",
      topk_ids.size(0));
  TORCH_CHECK(
      topk_weights.sizes() == topk_ids.sizes(),
      "topk_weights shape must match topk_ids ", topk_ids.sizes(),
      ", got ", topk_weights.sizes());

  nvfp4_validate_output_scale(w13_output_scale, "w13_output_scale");
  nvfp4_validate_output_scale(w2_output_scale, "w2_output_scale");
  const float w13_output_scale_f = static_cast<float>(w13_output_scale);
  const float w2_output_scale_f = static_cast<float>(w2_output_scale);
  const int64_t routes = M * T;
  TORCH_CHECK(
      w13_rows <= std::numeric_limits<int64_t>::max() / routes,
      "M*T*2I is too large");
  TORCH_CHECK(
      N <= std::numeric_limits<int64_t>::max() / M,
      "M*N is too large");

  std::vector<int64_t> route_ids(routes);
  if (topk_ids.scalar_type() == at::kInt) {
    const int32_t* ids = topk_ids.data_ptr<int32_t>();
    for (int64_t route = 0; route < routes; ++route) {
      TORCH_CHECK(
          ids[route] >= 0 && ids[route] < E,
          "topk_ids[", route / T, ", ", route % T, "]=", ids[route],
          " is outside [0, ", E, ")");
      route_ids[route] = ids[route];
    }
  } else {
    const int64_t* ids = topk_ids.data_ptr<int64_t>();
    for (int64_t route = 0; route < routes; ++route) {
      TORCH_CHECK(
          ids[route] >= 0 && ids[route] < E,
          "topk_ids[", route / T, ", ", route % T, "]=", ids[route],
          " is outside [0, ", E, ")");
      route_ids[route] = ids[route];
    }
  }

  const float* route_weights = topk_weights.data_ptr<float>();
  for (int64_t route = 0; route < routes; ++route) {
    TORCH_CHECK(
        std::isfinite(route_weights[route]),
        "topk_weights[", route / T, ", ", route % T,
        "] must be finite, got ", route_weights[route]);
  }

  const Nvfp4PreparedActivations prepared_x(
      x.data_ptr<c10::BFloat16>(), M, K, row_tile);
  const uint8_t* w13_data = w13.data_ptr<uint8_t>();
  const uint8_t* w13_scale_data = w13_scale.data_ptr<uint8_t>();
  const int64_t w13_packed_k = K / 2;
  const int64_t w13_scale_k = K / 16;
  auto hidden = torch::empty(
      {routes, I}, x.options().dtype(torch::kBFloat16));
  c10::BFloat16* hidden_data = hidden.data_ptr<c10::BFloat16>();

  // Keep gate and up reads contiguous within a block without adding a second
  // OpenMP region for a cache-resident elementwise SwiGLU pass.
  constexpr int64_t kW13RowBlock = 32;
  const int64_t stage1_work = routes * I;
  const int64_t stage1_grain = std::max<int64_t>(
      1, stage1_work /
          std::max<int64_t>(1, static_cast<int64_t>(at::get_num_threads())));
  if constexpr (kProfile) {
    profile->w13_backend = prepared_x.backend_name();
    profile->record(0);
  }
  at::parallel_for(0, stage1_work, stage1_grain, [&](int64_t begin, int64_t end) {
    std::array<float, kW13RowBlock> gate_values;
    int64_t position = begin;
    while (position < end) {
      const int64_t route = position / I;
      const int64_t intermediate_begin = position - route * I;
      const int64_t segment_end = std::min<int64_t>(end, (route + 1) * I);
      const int64_t intermediate_end = segment_end - route * I;
      const int64_t token = route / T;
      const int64_t expert = route_ids[route];
      const int64_t expert_row = expert * w13_rows;
      for (int64_t block_begin = intermediate_begin;
           block_begin < intermediate_end;
           block_begin += kW13RowBlock) {
        const int64_t block_end =
            std::min<int64_t>(intermediate_end, block_begin + kW13RowBlock);
        if (prepared_x.uses_row_tile()) {
          const int64_t row = expert_row + block_begin;
          const int64_t block_size = block_end - block_begin;
          std::array<float, kW13RowBlock> up_values;
          prepared_x.dot_rows(
              token, w13_data + row * w13_packed_k,
              w13_scale_data + row * w13_scale_k,
              block_size, gate_values.data());
          prepared_x.dot_rows(
              token, w13_data + (row + I) * w13_packed_k,
              w13_scale_data + (row + I) * w13_scale_k,
              block_size, up_values.data());
          for (int64_t i = 0; i < block_size; ++i) {
            const float gate = w13_output_scale_f * gate_values[i];
            const float up = w13_output_scale_f * up_values[i];
            hidden_data[route * I + block_begin + i] =
                c10::BFloat16(nvfp4_silu(gate) * up);
          }
          continue;
        }
        for (int64_t i = block_begin; i < block_end; ++i) {
          const int64_t row = expert_row + i;
          gate_values[i - block_begin] =
              w13_output_scale_f * prepared_x.dot(
                  token,
                  w13_data + row * w13_packed_k,
                  w13_scale_data + row * w13_scale_k);
        }
        for (int64_t i = block_begin; i < block_end; ++i) {
          const int64_t row = expert_row + I + i;
          const float up = w13_output_scale_f * prepared_x.dot(
              token,
              w13_data + row * w13_packed_k,
              w13_scale_data + row * w13_scale_k);
          hidden_data[route * I + i] = c10::BFloat16(
              nvfp4_silu(gate_values[i - block_begin]) * up);
        }
      }
      position = segment_end;
    }
  });
  if constexpr (kProfile) {
    profile->record(1);
  }

  // Returning from the W13 region is the only required stage barrier.
  const Nvfp4PreparedActivations prepared_hidden(hidden_data, routes, I, row_tile);
  if constexpr (kProfile) {
    profile->w2_backend = prepared_hidden.backend_name();
    profile->record(2);
  }
  const uint8_t* w2_data = w2.data_ptr<uint8_t>();
  const uint8_t* w2_scale_data = w2_scale.data_ptr<uint8_t>();
  const int64_t w2_packed_k = I / 2;
  const int64_t w2_scale_k = I / 16;
  auto output = torch::empty(
      {M, N}, x.options().dtype(torch::kBFloat16));
  c10::BFloat16* output_data = output.data_ptr<c10::BFloat16>();

  const int64_t stage2_work = M * N;
  const int64_t stage2_grain = std::max<int64_t>(
      1, stage2_work /
          std::max<int64_t>(1, static_cast<int64_t>(at::get_num_threads())));
  constexpr int64_t kW2OutputBlock = 256;
  if constexpr (kProfile) {
    profile->record(3);
  }
  at::parallel_for(0, stage2_work, stage2_grain, [&](int64_t begin, int64_t end) {
    std::array<float, kW2OutputBlock> accumulator;
    std::array<float, kW2OutputBlock> projections;
    int64_t position = begin;
    while (position < end) {
      const int64_t token = position / N;
      const int64_t output_begin = position - token * N;
      const int64_t segment_end = std::min<int64_t>(end, (token + 1) * N);
      const int64_t output_end = segment_end - token * N;

      for (int64_t block_begin = output_begin;
           block_begin < output_end;
           block_begin += kW2OutputBlock) {
        const int64_t block_end =
            std::min<int64_t>(output_end, block_begin + kW2OutputBlock);
        std::fill(
            accumulator.begin(),
            accumulator.begin() + (block_end - block_begin),
            0.0f);
        for (int64_t topk = 0; topk < T; ++topk) {
          const int64_t route = token * T + topk;
          const int64_t expert = route_ids[route];
          const float routing_weight = route_weights[route];
          const int64_t expert_row = expert * N;
          if (prepared_hidden.uses_row_tile()) {
            const int64_t row = expert_row + block_begin;
            prepared_hidden.dot_rows(
                route, w2_data + row * w2_packed_k,
                w2_scale_data + row * w2_scale_k,
                block_end - block_begin, projections.data());
            for (int64_t n = block_begin; n < block_end; ++n) {
              const float projected =
                  w2_output_scale_f * projections[n - block_begin];
              accumulator[n - block_begin] += routing_weight * projected;
            }
            continue;
          }
          for (int64_t n = block_begin; n < block_end; ++n) {
            const int64_t row = expert_row + n;
            const float projected = w2_output_scale_f * prepared_hidden.dot(
                route,
                w2_data + row * w2_packed_k,
                w2_scale_data + row * w2_scale_k);
            accumulator[n - block_begin] += routing_weight * projected;
          }
        }
        for (int64_t n = block_begin; n < block_end; ++n) {
          output_data[token * N + n] =
              c10::BFloat16(accumulator[n - block_begin]);
        }
      }
      position = segment_end;
    }
  });
  if constexpr (kProfile) {
    profile->record(4);
  }
  return output;
}

inline torch::Tensor fused_nvfp4_moe_swiglu_cpu(
    const torch::Tensor& x,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    double w13_output_scale,
    double w2_output_scale) {
  return fused_nvfp4_moe_swiglu_cpu_impl<false>(
      x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights,
      w13_output_scale, w2_output_scale, nullptr);
}

inline std::tuple<torch::Tensor, std::vector<double>, std::string, std::string>
fused_nvfp4_moe_swiglu_cpu_profile(
    const torch::Tensor& x,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    double w13_output_scale,
    double w2_output_scale) {
  Nvfp4MoeProfile profile;
  auto output = fused_nvfp4_moe_swiglu_cpu_impl<true>(
      x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights,
      w13_output_scale, w2_output_scale, &profile);
  profile.record(5);
  return {
      output, std::vector<double>(profile.seconds.begin(), profile.seconds.end()),
      profile.w13_backend, profile.w2_backend};
}

} // namespace fused_moe_nvfp4
} // namespace tutel
#undef TUTEL_NVFP4_AVX2_AVAILABLE
#undef TUTEL_NVFP4_AVX2_TARGET
#undef TUTEL_NVFP4_AVX512_BF16_AVAILABLE
#undef TUTEL_NVFP4_AVX512_BF16_TARGET
