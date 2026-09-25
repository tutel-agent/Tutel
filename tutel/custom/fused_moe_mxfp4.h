// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace tutel {
namespace fused_moe_mxfp4 {

constexpr std::array<float, 16> kMxfp4E2m1 = {{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
}};

alignas(64) constexpr std::array<uint16_t, 32> kMxfp4E2m1Bf16 = {{
    0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
    0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0,
    0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
    0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0,
}};

inline float mxfp4_e8m0_to_float(uint8_t exponent) {
  const uint32_t bits =
      exponent == 0 ? UINT32_C(0x00400000) :
      static_cast<uint32_t>(exponent) << 23;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline const std::array<float, 256>& mxfp4_e8m0_lut() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> result{};
    for (int i = 0; i < 256; ++i) {
      result[i] = mxfp4_e8m0_to_float(static_cast<uint8_t>(i));
    }
    return result;
  }();
  return table;
}

inline float mxfp4_dot_scalar(
    const float* activation_even,
    const float* activation_odd,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    int64_t packed_k) {
  const auto& scale_lut = mxfp4_e8m0_lut();
  float sum = 0.0f;
  for (int64_t p = 0; p < packed_k; ++p) {
    const uint8_t packed = packed_weight[p];
    const float block_scale = scale_lut[scale[p / 16]];
    sum += activation_even[p] *
            (kMxfp4E2m1[packed & 0x0f] * block_scale) +
        activation_odd[p] *
            (kMxfp4E2m1[packed >> 4] * block_scale);
  }
  return sum;
}

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#define TUTEL_MXFP4_AVX2_AVAILABLE 1
#define TUTEL_MXFP4_AVX2_TARGET __attribute__((target("avx2,fma")))
#elif (defined(_M_X64) || defined(_M_IX86)) && \
    (defined(__AVX2__) || defined(_M_AVX2))
#define TUTEL_MXFP4_AVX2_AVAILABLE 1
#define TUTEL_MXFP4_AVX2_TARGET
#else
#define TUTEL_MXFP4_AVX2_AVAILABLE 0
#define TUTEL_MXFP4_AVX2_TARGET
#endif

#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__) && \
    ((defined(__clang__) && __clang_major__ >= 12) || \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11))
#define TUTEL_MXFP4_AVX512_BF16_AVAILABLE 1
#define TUTEL_MXFP4_AVX512_BF16_TARGET \
  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16,fma")))
#else
#define TUTEL_MXFP4_AVX512_BF16_AVAILABLE 0
#define TUTEL_MXFP4_AVX512_BF16_TARGET
#endif

#if TUTEL_MXFP4_AVX512_BF16_AVAILABLE
inline bool mxfp4_cpu_supports_avx512_bf16() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512vl") &&
      __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512bf16") &&
      __builtin_cpu_supports("fma");
}

TUTEL_MXFP4_AVX512_BF16_TARGET inline float mxfp4_dot_avx512_bf16(
    const c10::BFloat16* activation_permuted,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    int64_t packed_k) {
  const __m256i nibble_mask = _mm256_set1_epi16(0x0f);
  const __m512i e2m1_lut = _mm512_load_si512(kMxfp4E2m1Bf16.data());
  const auto& scale_lut = mxfp4_e8m0_lut();
  __m512 sum = _mm512_setzero_ps();

  for (int64_t p = 0; p < packed_k; p += 16) {
    const __m128i packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_weight + p));
    const __m256i packed_16 = _mm256_cvtepu8_epi16(packed);
    const __m256i even_indices = _mm256_and_si256(packed_16, nibble_mask);
    const __m256i odd_indices = _mm256_srli_epi16(packed_16, 4);
    __m512i indices = _mm512_castsi256_si512(even_indices);
    indices = _mm512_inserti64x4(indices, odd_indices, 1);
    const __m512i weight_bits = _mm512_permutexvar_epi16(indices, e2m1_lut);
    const __m512i activation_bits = _mm512_loadu_si512(
        reinterpret_cast<const void*>(activation_permuted + 2 * p));
    const __m512 contribution = _mm512_dpbf16_ps(
        _mm512_setzero_ps(),
        (__m512bh)activation_bits,
        (__m512bh)weight_bits);
    const __m512 block_scale =
        _mm512_set1_ps(scale_lut[scale[p / 16]]);
    sum = _mm512_fmadd_ps(contribution, block_scale, sum);
  }
  return _mm512_reduce_add_ps(sum);
}
#endif

#if TUTEL_MXFP4_AVX2_AVAILABLE
inline bool mxfp4_cpu_supports_avx2_fma() {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") &&
      __builtin_cpu_supports("fma");
#else
  return true;
#endif
}

TUTEL_MXFP4_AVX2_TARGET inline float mxfp4_dot_avx2(
    const float* activation_even,
    const float* activation_odd,
    const uint8_t* packed_weight,
    const uint8_t* scale,
    int64_t packed_k) {
  const __m128i nibble_mask = _mm_set1_epi8(0x0f);
  const __m128i e2m1_lut = _mm_setr_epi8(
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const auto& scale_lut = mxfp4_e8m0_lut();
  __m256 sum = _mm256_setzero_ps();

  int64_t p = 0;
  for (; p + 16 <= packed_k; p += 16) {
    const __m128i packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_weight + p));
    const __m128i low =
        _mm_shuffle_epi8(e2m1_lut, _mm_and_si128(packed, nibble_mask));
    const __m128i high = _mm_shuffle_epi8(
        e2m1_lut,
        _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask));
    const __m256 low0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low));
    const __m256 low1 =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low, 8)));
    const __m256 high0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high));
    const __m256 high1 =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high, 8)));
    const __m256 block_scale =
        _mm256_set1_ps(scale_lut[scale[p / 16]] * 0.5f);
    sum = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_even + p),
        _mm256_mul_ps(low0, block_scale),
        sum);
    sum = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_odd + p),
        _mm256_mul_ps(high0, block_scale),
        sum);
    sum = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_even + p + 8),
        _mm256_mul_ps(low1, block_scale),
        sum);
    sum = _mm256_fmadd_ps(
        _mm256_loadu_ps(activation_odd + p + 8),
        _mm256_mul_ps(high1, block_scale),
        sum);
  }

  const __m128 halves = _mm_add_ps(
      _mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
  const __m128 pairs =
      _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
  float result =
      _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
  _mm256_zeroupper();
  for (; p < packed_k; ++p) {
    const uint8_t packed = packed_weight[p];
    const float block_scale = scale_lut[scale[p / 16]];
    result += activation_even[p] *
            (kMxfp4E2m1[packed & 0x0f] * block_scale) +
        activation_odd[p] *
            (kMxfp4E2m1[packed >> 4] * block_scale);
  }
  return result;
}
#endif

inline float mxfp4_sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-value));
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0f + exp_value);
}

inline float mxfp4_situ_glu(float gate, float up) {
  constexpr float kBeta = 4.0f;
  constexpr float kBetaLin = 25.0f;
  const float gate_term =
      kBeta * std::tanh(gate / kBeta) * mxfp4_sigmoid(gate);
  const float up_term = kBetaLin * std::tanh(up / kBetaLin);
  return gate_term * up_term;
}

struct Mxfp4PreparedActivations {
  int64_t rows;
  int64_t K;
  int64_t packed_k;
#if TUTEL_MXFP4_AVX512_BF16_AVAILABLE
  bool use_avx512_bf16;
  std::vector<c10::BFloat16> avx512_bf16;
#endif
#if TUTEL_MXFP4_AVX2_AVAILABLE
  bool use_avx2;
#endif
  std::vector<float> even;
  std::vector<float> odd;
  std::vector<float> max_abs;

  Mxfp4PreparedActivations(
      const c10::BFloat16* activation,
      int64_t activation_rows,
      int64_t activation_k)
      : rows(activation_rows), K(activation_k), packed_k(activation_k / 2)
#if TUTEL_MXFP4_AVX512_BF16_AVAILABLE
      , use_avx512_bf16(
          activation_k % 32 == 0 && mxfp4_cpu_supports_avx512_bf16())
#endif
#if TUTEL_MXFP4_AVX2_AVAILABLE
      , use_avx2(mxfp4_cpu_supports_avx2_fma())
#endif
      , even(rows * packed_k), odd(rows * packed_k), max_abs(rows, 0.0f) {
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t source_offset = row * K;
      const int64_t target_offset = row * packed_k;
      float row_max = 0.0f;
      for (int64_t p = 0; p < packed_k; ++p) {
        const float even_value =
            static_cast<float>(activation[source_offset + 2 * p]);
        const float odd_value =
            static_cast<float>(activation[source_offset + 2 * p + 1]);
        even[target_offset + p] = even_value;
        odd[target_offset + p] = odd_value;
        if (!std::isfinite(even_value) || !std::isfinite(odd_value)) {
          row_max = std::numeric_limits<float>::infinity();
        } else {
          row_max = std::max(
              row_max, std::max(std::abs(even_value), std::abs(odd_value)));
        }
      }
      max_abs[row] = row_max;
    }
#if TUTEL_MXFP4_AVX512_BF16_AVAILABLE
    if (use_avx512_bf16) {
      avx512_bf16.resize(rows * K);
      for (int64_t row = 0; row < rows; ++row) {
        const int64_t row_offset = row * K;
        for (int64_t k = 0; k < K; k += 32) {
          for (int64_t i = 0; i < 16; ++i) {
            avx512_bf16[row_offset + k + i] =
                activation[row_offset + k + 2 * i];
            avx512_bf16[row_offset + k + 16 + i] =
                activation[row_offset + k + 2 * i + 1];
          }
        }
      }
    }
#endif
  }

  inline bool vector_safe(
      int64_t row, const uint8_t* scale) const {
    if (!std::isfinite(max_abs[row])) {
      return false;
    }
    if (static_cast<double>(max_abs[row]) * 12.0 >
        static_cast<double>(std::numeric_limits<float>::max())) {
      return false;
    }
    uint8_t maximum_exponent = 0;
    for (int64_t group = 0; group < K / 32; ++group) {
      maximum_exponent = std::max(maximum_exponent, scale[group]);
    }
    // Scalar semantics form FP4 * scale before multiplying activation.
    // E8M0 exponents 253+ can overflow that product for the largest FP4 code.
    if (maximum_exponent >= 253) {
      return false;
    }
    const double absolute_bound =
        static_cast<double>(max_abs[row]) * 6.0 *
        static_cast<double>(mxfp4_e8m0_lut()[maximum_exponent]) *
        static_cast<double>(K);
    return absolute_bound <=
        static_cast<double>(std::numeric_limits<float>::max());
  }

  inline float dot(
      int64_t row,
      const uint8_t* weight,
      const uint8_t* scale) const {
    if (vector_safe(row, scale)) {
#if TUTEL_MXFP4_AVX512_BF16_AVAILABLE
      if (use_avx512_bf16) {
        return mxfp4_dot_avx512_bf16(
            avx512_bf16.data() + row * K, weight, scale, packed_k);
      }
#endif
#if TUTEL_MXFP4_AVX2_AVAILABLE
      if (use_avx2) {
        return mxfp4_dot_avx2(
            even.data() + row * packed_k,
            odd.data() + row * packed_k,
            weight,
            scale,
            packed_k);
      }
#endif
    }
    return mxfp4_dot_scalar(
        even.data() + row * packed_k,
        odd.data() + row * packed_k,
        weight,
        scale,
        packed_k);
  }
};

inline torch::Tensor fused_mxfp4_moe_situ_cpu(
    const torch::Tensor& x,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights) {
  constexpr const char* op = "tutel_ops::fused_mxfp4_moe_situ_cpu";
  TORCH_CHECK(x.device().is_cpu(), op, ": x must be a CPU tensor");
  TORCH_CHECK(w13.device().is_cpu(), op, ": w13 must be a CPU tensor");
  TORCH_CHECK(
      w13_scale.device().is_cpu(), op, ": w13_scale must be a CPU tensor");
  TORCH_CHECK(w2.device().is_cpu(), op, ": w2 must be a CPU tensor");
  TORCH_CHECK(
      w2_scale.device().is_cpu(), op, ": w2_scale must be a CPU tensor");
  TORCH_CHECK(
      topk_ids.device().is_cpu(), op, ": topk_ids must be a CPU tensor");
  TORCH_CHECK(
      topk_weights.device().is_cpu(),
      op,
      ": topk_weights must be a CPU tensor");
  TORCH_CHECK(x.is_contiguous(), op, ": x must be contiguous");
  TORCH_CHECK(w13.is_contiguous(), op, ": w13 must be contiguous");
  TORCH_CHECK(
      w13_scale.is_contiguous(), op, ": w13_scale must be contiguous");
  TORCH_CHECK(w2.is_contiguous(), op, ": w2 must be contiguous");
  TORCH_CHECK(w2_scale.is_contiguous(), op, ": w2_scale must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), op, ": topk_ids must be contiguous");
  TORCH_CHECK(
      topk_weights.is_contiguous(), op, ": topk_weights must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16,
      op,
      ": x must have dtype torch.bfloat16");
  TORCH_CHECK(
      w13.scalar_type() == at::kByte,
      op,
      ": w13 must have dtype torch.uint8");
  TORCH_CHECK(
      w13_scale.scalar_type() == at::kByte,
      op,
      ": w13_scale must have dtype torch.uint8");
  TORCH_CHECK(
      w2.scalar_type() == at::kByte,
      op,
      ": w2 must have dtype torch.uint8");
  TORCH_CHECK(
      w2_scale.scalar_type() == at::kByte,
      op,
      ": w2_scale must have dtype torch.uint8");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt ||
          topk_ids.scalar_type() == at::kLong,
      op,
      ": topk_ids must have dtype torch.int32 or torch.int64");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      op,
      ": topk_weights must have dtype torch.float32");
  TORCH_CHECK(x.dim() == 2, op, ": x must have shape [M, K], got ", x.sizes());
  TORCH_CHECK(
      w13.dim() == 3,
      op,
      ": w13 must have shape [E, 2*I, K/2], got ",
      w13.sizes());
  TORCH_CHECK(
      w13_scale.dim() == 3,
      op,
      ": w13_scale must have shape [E, 2*I, K/32], got ",
      w13_scale.sizes());
  TORCH_CHECK(
      w2.dim() == 3,
      op,
      ": w2 must have shape [E, N, I/2], got ",
      w2.sizes());
  TORCH_CHECK(
      w2_scale.dim() == 3,
      op,
      ": w2_scale must have shape [E, N, I/32], got ",
      w2_scale.sizes());
  TORCH_CHECK(
      topk_ids.dim() == 2,
      op,
      ": topk_ids must have shape [M, T], got ",
      topk_ids.sizes());
  TORCH_CHECK(
      topk_weights.dim() == 2,
      op,
      ": topk_weights must have shape [M, T], got ",
      topk_weights.sizes());

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t E = w13.size(0);
  const int64_t w13_rows = w13.size(1);
  const int64_t I = w13_rows / 2;
  const int64_t N = w2.size(1);
  const int64_t T = topk_ids.size(1);
  TORCH_CHECK(M > 0, op, ": x dimension M must be positive");
  TORCH_CHECK(
      K > 0 && K % 32 == 0,
      op,
      ": x dimension K must be positive and divisible by 32, got ",
      K);
  TORCH_CHECK(E > 0, op, ": expert dimension E must be positive");
  TORCH_CHECK(
      w13_rows > 0 && w13_rows % 2 == 0,
      op,
      ": w13 dimension 1 must be positive and even, got ",
      w13_rows);
  TORCH_CHECK(
      I % 32 == 0,
      op,
      ": intermediate dimension I must be divisible by 32, got ",
      I);
  TORCH_CHECK(N > 0, op, ": w2 output dimension N must be positive");
  TORCH_CHECK(T > 0, op, ": top-k dimension T must be positive");
  TORCH_CHECK(
      T <= std::numeric_limits<int64_t>::max() / M,
      op,
      ": M*T is too large");
  const int64_t routes = M * T;
  TORCH_CHECK(
      w13_rows <= std::numeric_limits<int64_t>::max() / routes,
      op,
      ": M*T*2I is too large");
  TORCH_CHECK(
      N <= std::numeric_limits<int64_t>::max() / M,
      op,
      ": M*N is too large");
  TORCH_CHECK(
      w13.size(2) == K / 2,
      op,
      ": w13 dimension 2 must be K/2=",
      K / 2,
      ", got ",
      w13.size(2));
  TORCH_CHECK(
      w13_scale.size(0) == E &&
          w13_scale.size(1) == w13_rows &&
          w13_scale.size(2) == K / 32,
      op,
      ": w13_scale must have shape [",
      E,
      ", ",
      w13_rows,
      ", ",
      K / 32,
      "], got ",
      w13_scale.sizes());
  TORCH_CHECK(
      w2.size(0) == E,
      op,
      ": w2 expert dimension must match w13 E=",
      E,
      ", got ",
      w2.size(0));
  TORCH_CHECK(
      w2.size(2) == I / 2,
      op,
      ": w2 dimension 2 must be I/2=",
      I / 2,
      ", got ",
      w2.size(2));
  TORCH_CHECK(
      w2_scale.size(0) == E &&
          w2_scale.size(1) == N &&
          w2_scale.size(2) == I / 32,
      op,
      ": w2_scale must have shape [",
      E,
      ", ",
      N,
      ", ",
      I / 32,
      "], got ",
      w2_scale.sizes());
  TORCH_CHECK(
      topk_ids.size(0) == M,
      op,
      ": topk_ids dimension 0 must match x M=",
      M,
      ", got ",
      topk_ids.size(0));
  TORCH_CHECK(
      topk_weights.sizes() == topk_ids.sizes(),
      op,
      ": topk_weights shape must match topk_ids ",
      topk_ids.sizes(),
      ", got ",
      topk_weights.sizes());

  std::vector<int64_t> route_ids(routes);
  if (topk_ids.scalar_type() == at::kInt) {
    const int32_t* ids = topk_ids.data_ptr<int32_t>();
    for (int64_t route = 0; route < routes; ++route) {
      TORCH_CHECK(
          ids[route] >= 0 && ids[route] < E,
          op,
          ": topk_ids[",
          route / T,
          ", ",
          route % T,
          "]=",
          ids[route],
          " is outside [0, ",
          E,
          ")");
      route_ids[route] = ids[route];
    }
  } else {
    const int64_t* ids = topk_ids.data_ptr<int64_t>();
    for (int64_t route = 0; route < routes; ++route) {
      TORCH_CHECK(
          ids[route] >= 0 && ids[route] < E,
          op,
          ": topk_ids[",
          route / T,
          ", ",
          route % T,
          "]=",
          ids[route],
          " is outside [0, ",
          E,
          ")");
      route_ids[route] = ids[route];
    }
  }
  const float* route_weights = topk_weights.data_ptr<float>();
  for (int64_t route = 0; route < routes; ++route) {
    TORCH_CHECK(
        std::isfinite(route_weights[route]),
        op,
        ": topk_weights[",
        route / T,
        ", ",
        route % T,
        "] must be finite, got ",
        route_weights[route]);
  }

  const Mxfp4PreparedActivations prepared_x(
      x.data_ptr<c10::BFloat16>(), M, K);
  const uint8_t* w13_data = w13.data_ptr<uint8_t>();
  const uint8_t* w13_scale_data = w13_scale.data_ptr<uint8_t>();
  const int64_t w13_packed_k = K / 2;
  const int64_t w13_scale_k = K / 32;
  auto hidden = torch::empty(
      {routes, I}, x.options().dtype(torch::kBFloat16));
  c10::BFloat16* hidden_data = hidden.data_ptr<c10::BFloat16>();

  constexpr int64_t kW13RowBlock = 32;
  const int64_t stage1_work = routes * I;
  const int64_t stage1_grain = std::max<int64_t>(
      1,
      stage1_work /
          std::max<int64_t>(
              1, static_cast<int64_t>(at::get_num_threads())));
  at::parallel_for(
      0,
      stage1_work,
      stage1_grain,
      [&](int64_t begin, int64_t end) {
        std::array<float, kW13RowBlock> gate_values;
        int64_t position = begin;
        while (position < end) {
          const int64_t route = position / I;
          const int64_t intermediate_begin = position - route * I;
          const int64_t segment_end =
              std::min<int64_t>(end, (route + 1) * I);
          const int64_t intermediate_end = segment_end - route * I;
          const int64_t token = route / T;
          const int64_t expert = route_ids[route];
          const int64_t expert_row = expert * w13_rows;
          for (int64_t block_begin = intermediate_begin;
               block_begin < intermediate_end;
               block_begin += kW13RowBlock) {
            const int64_t block_end = std::min<int64_t>(
                intermediate_end, block_begin + kW13RowBlock);
            for (int64_t i = block_begin; i < block_end; ++i) {
              const int64_t row = expert_row + i;
              gate_values[i - block_begin] = prepared_x.dot(
                  token,
                  w13_data + row * w13_packed_k,
                  w13_scale_data + row * w13_scale_k);
            }
            for (int64_t i = block_begin; i < block_end; ++i) {
              const int64_t row = expert_row + I + i;
              const float up = prepared_x.dot(
                  token,
                  w13_data + row * w13_packed_k,
                  w13_scale_data + row * w13_scale_k);
              hidden_data[route * I + i] = c10::BFloat16(
                  mxfp4_situ_glu(gate_values[i - block_begin], up));
            }
          }
          position = segment_end;
        }
      });

  const Mxfp4PreparedActivations prepared_hidden(hidden_data, routes, I);
  const uint8_t* w2_data = w2.data_ptr<uint8_t>();
  const uint8_t* w2_scale_data = w2_scale.data_ptr<uint8_t>();
  const int64_t w2_packed_k = I / 2;
  const int64_t w2_scale_k = I / 32;
  auto output = torch::empty(
      {M, N}, x.options().dtype(torch::kBFloat16));
  c10::BFloat16* output_data = output.data_ptr<c10::BFloat16>();

  constexpr int64_t kW2OutputBlock = 256;
  const int64_t stage2_work = M * N;
  const int64_t stage2_grain = std::max<int64_t>(
      1,
      stage2_work /
          std::max<int64_t>(
              1, static_cast<int64_t>(at::get_num_threads())));
  at::parallel_for(
      0,
      stage2_work,
      stage2_grain,
      [&](int64_t begin, int64_t end) {
        std::array<float, kW2OutputBlock> accumulator;
        int64_t position = begin;
        while (position < end) {
          const int64_t token = position / N;
          const int64_t output_begin = position - token * N;
          const int64_t segment_end =
              std::min<int64_t>(end, (token + 1) * N);
          const int64_t output_end = segment_end - token * N;
          for (int64_t block_begin = output_begin;
               block_begin < output_end;
               block_begin += kW2OutputBlock) {
            const int64_t block_end = std::min<int64_t>(
                output_end, block_begin + kW2OutputBlock);
            std::fill(
                accumulator.begin(),
                accumulator.begin() + (block_end - block_begin),
                0.0f);
            for (int64_t topk = 0; topk < T; ++topk) {
              const int64_t route = token * T + topk;
              const int64_t expert = route_ids[route];
              const float routing_weight = route_weights[route];
              const int64_t expert_row = expert * N;
              for (int64_t n = block_begin; n < block_end; ++n) {
                const int64_t row = expert_row + n;
                const float projected = prepared_hidden.dot(
                    route,
                    w2_data + row * w2_packed_k,
                    w2_scale_data + row * w2_scale_k);
                accumulator[n - block_begin] +=
                    routing_weight * projected;
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
  return output;
}

} // namespace fused_moe_mxfp4
} // namespace tutel

#undef TUTEL_MXFP4_AVX2_AVAILABLE
#undef TUTEL_MXFP4_AVX2_TARGET
#undef TUTEL_MXFP4_AVX512_BF16_AVAILABLE
#undef TUTEL_MXFP4_AVX512_BF16_TARGET
