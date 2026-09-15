#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <immintrin.h>
#include <sched.h>
#include <chrono>
#include <ATen/record_function.h>

// BF16 storage is widened exactly. Products and reductions use FP32;
// only the completed dot product is rounded back to BF16 (round to nearest).
static inline __m256 widen8(const at::BFloat16* p) {
  const auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

static inline float sum8(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(_mm_hadd_ps(s, s));
}

at::Tensor bf16_gemv(const at::Tensor& w, const at::Tensor& x) {
  TORCH_CHECK(w.device().is_cpu() && x.device().is_cpu(), "CPU tensors required");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16,
              "BF16 tensors required");
  TORCH_CHECK(w.dim() == 2 && x.dim() == 1 && w.size(1) == x.numel(), "GEMV shape mismatch");
  TORCH_CHECK(w.is_contiguous() && x.is_contiguous(), "contiguous tensors required");
  const auto m = w.size(0), n = w.size(1);
  auto y = at::empty({m}, x.options());
  auto xf = x.to(at::kFloat);
  const auto* xp = xf.data_ptr<float>();
  const auto* wp = w.data_ptr<at::BFloat16>();
  auto* yp = y.data_ptr<at::BFloat16>();
  at::parallel_for(0, (m + 3) / 4, 1, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      const auto r = b * 4;
      const auto nr = std::min<int64_t>(4, m - r);
      __m256 a[4], c[4];
      for (int k = 0; k < 4; ++k) a[k] = c[k] = _mm256_setzero_ps();
      int64_t j = 0;
      for (; j + 15 < n; j += 16) {
        const auto x0 = _mm256_loadu_ps(xp + j);
        const auto x1 = _mm256_loadu_ps(xp + j + 8);
        for (int k = 0; k < nr; ++k) {
          const auto* p = wp + (r + k) * n + j;
          a[k] = _mm256_fmadd_ps(widen8(p), x0, a[k]);
          c[k] = _mm256_fmadd_ps(widen8(p + 8), x1, c[k]);
        }
      }
      for (int k = 0; k < nr; ++k) {
        float s = sum8(_mm256_add_ps(a[k], c[k]));
        for (int64_t tail = j; tail < n; ++tail)
          s += float(wp[(r + k) * n + tail]) * xp[tail];
        yp[r + k] = at::BFloat16(s);
      }
    }
  });
  return y;
}

static void bind_threads(c10::List<int64_t> cpus) {
  TORCH_CHECK(cpus.size() == size_t(at::get_num_threads()), "one CPU per intra-op worker required");
  at::parallel_for(0, cpus.size(), 1, [&](int64_t begin, int64_t end) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    auto cpu = cpus.get(at::get_thread_num());
    TORCH_CHECK(cpu >= 0 && cpu < CPU_SETSIZE, "invalid CPU id");
    CPU_SET(cpu, &mask);
    TORCH_CHECK(sched_setaffinity(0, sizeof(mask), &mask) == 0, "worker affinity failed");
  });
}

static std::tuple<at::Tensor,at::Tensor> bf16_experts(
    c10::List<at::Tensor> packs, const at::Tensor& x, int64_t intermediate) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "contiguous CPU BF16 activation required");
  const auto h = x.numel(), i = intermediate;
  TORCH_CHECK(i > 0 && h > 0, "positive expert dimensions required");
  auto times = at::empty({int64_t(packs.size())}, x.options().dtype(at::kDouble));
  auto* ms = times.data_ptr<double>();
  std::vector<at::Tensor> outputs;
  auto vector = x.reshape({h});
  for (size_t e = 0; e < packs.size(); ++e) {
    RECORD_FUNCTION("SMoE::cpu_expert_native", std::vector<c10::IValue>());
    const auto pack = packs.get(e);
    TORCH_CHECK(pack.device().is_cpu() && pack.scalar_type() == at::kBFloat16
                && pack.is_contiguous() && pack.numel() == 3*h*i, "invalid BF16 expert pack");
    const auto begin = std::chrono::steady_clock::now();
    auto gu = at::mv(pack.narrow(0,0,2*h*i).view({2*i,h}), vector);
    auto mid = at::mul(at::silu(gu.narrow(0,0,i)), gu.narrow(0,i,i));
    auto out = at::mv(pack.narrow(0,2*h*i,h*i).view({h,i}), mid);
    ms[e] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    outputs.push_back(out);
  }
  return {outputs.empty() ? at::empty({0,h},x.options()) : at::stack(outputs),times};
}

TORCH_LIBRARY(smoe_cpu, m) {
  m.def("bf16_gemv(Tensor w, Tensor x) -> Tensor");
  m.def("bind_threads(int[] cpus) -> ()", bind_threads);
  m.def("bf16_experts(Tensor[] packs, Tensor x, int intermediate) -> (Tensor, Tensor)", bf16_experts);
}
TORCH_LIBRARY_IMPL(smoe_cpu, CPU, m) { m.impl("bf16_gemv", bf16_gemv); }
