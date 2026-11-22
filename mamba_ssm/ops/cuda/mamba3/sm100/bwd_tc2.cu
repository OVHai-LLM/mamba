// Mamba-3 MIMO varlen backward, Blackwell tcgen05 edition (sm_100 / sm_103), Olala configuration: R=4, N=128,
// P=64, 32 rotary phases, one Q/K group, reduceO + Z gate + D skip, bf16 activations.
//
// Chunk-parallel backward over 32-token sub-blocks (128 MIMO rows, row = token * R + rank). With, per head,
// qhat / khat = rotated (q / k + bias) rows, PsiV = v * mimo_v, dO = dy * mimo_o * silu(z * mimo_z),
// l = cumsum of ADT within the sub-block (inclusive), f = gamma + shifted gamma, W[t][s] = exp(l_t - l_s) f_s (s < t),
// gamma_t (s = t), h_b = state entering sub-block b, D_b = gradient of the loss w.r.t. the state leaving it:
//
//   MODE_LOCAL  Sloc_b = khat^T (PsiV * exp(l_end - l) f),  dloc_b = qhat^T (dO * exp(l))       -> h, D (Triton scan)
//               (epilogue one item late, bf16 tiles staged in smem and written by TMA stores)
//   MODE_FUSED  per (sub-block, head), both sides on the same operands:
//     q side  (rows t):  G = qhat khat^T, dP = dO PsiV^T;  O1 = qhat h_b, O2 = (G * W) PsiV,
//             dqhat = (dO * exp(l)) h_b^T + (dP * W) khat
//             -> o = exp(l) O1 + O2 + D PsiV -> the gate gradients (dz, dmimo_o, dmimo_z); dq (rotation undone,
//                summed over the heads); the q part of the phase gradient; sum_{s<t} W A + exp(l_t) O1 . dO
//                (A = G * dP) and the diagonal (gamma) gradient sum_{s=t} A
//     kv side (rows s):  GT = khat qhat^T, dPT = PsiV dO^T;  kD = khat D_b, dPsi2 = (GT * W^T) dO,
//             dkhat = (PsiV * exp(l_end - l) f) D_b^T + (dPT * W^T) qhat
//             -> dPsiV = exp(l_end - l) f kD + dPsi2 + D dO -> dv, dmimo_v, dD; dk (summed over the heads); the k
//                part of the phase gradient; the trapezoid factor gradient sum_{t>s} exp(l_t - l_s) A
//                + exp(l_end - l_s) kD . PsiV
// The masked matrices go back to tensor memory as bf16 (over their own fp32 columns) and feed the second products as
// the A operand, so the transposed products are computed directly (no column reductions, no shared-memory staging
// of 128 x 128 tiles). The kv side also yields <h_{b+1}, D_b> = exp(tot) <h_b, D_b> + <Sloc_b, D_b> for dADT.
// Per-head reductions (bias, mimo and D gradients) are summed in a fixed order into per-(CTA, head) accumulators
// owned by the CTA (deterministic).
//
// One persistent CTA per SM (16 warps), work = sub-blocks (CTA-strided), the heads of a sub-block one after the other
// (the Q / K rows of a sub-block are read from L2 for every head). Per (sub-block, head): TMA of the raw rows (Q, K,
// V, dy, z, phases, both state tiles; the next head's rows are prefetched as soon as their buffers free up), all
// warps build the bf16 operands in place (bias + rotation, PsiV, dO and the scaled copies, token tables), one thread
// in each of warps 0 / 4 / 8 / 12 issues an independent UMMA chain (M = 128, fp32 accumulators in TMEM; G and O1 start
// as soon as qhat / khat are built, the kv side's first products as soon as the q side has read its columns), all
// warps run the epilogues (warp w: TMEM lanes 32 (w % 4), column group w / 4; reductions over lanes are
// reduce-scatters). The kernel is bound by the CUDA-core epilogue / operand work, not by the tensor cores.
// V2 of the FUSED kernel of bwd_tc.cu (M3_BWD_V2=1): the masked G * W and dP * W go to shared memory as bf16
// K-major tiles; the q side reads them as K-major A operands and the kv side as MN-major A operands (the transposed
// products), so the kv side needs neither GT / dPT products nor masks, and all its products are issued right after
// the q masks (they run under the q-side epilogues). The state-part A operands dO exp(l) / PsiV exp(l_end - l) f
// live in tensor memory (their shared-memory tiles hold the masked tiles). V / dy / z / phases single-buffered.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32, C = 32, ROWS = C * R;
constexpr int THREADS = 512, QC = THREADS / 128;  // QC column groups in the epilogues
constexpr int TILE = ROWS * 64;   // one 128-row x 64-col bf16 tile (128B rows, SW128)
constexpr int TILE_B = TILE * 2;  // 16 KB
constexpr int CP = C + 1, NQP = NQ + 1, PP = P + 8;  // padded strides (bank-conflict-free columns / ranks)
constexpr int MODE_LOCAL = 0, MODE_FUSED = 1, SIDE_Q = 0, SIDE_KV = 1;
// tensor memory columns
// G | dP (fp32; the masked bf16 G*W / dP*W are written back over their first 64 columns) | O1 | O2 | dq (dk)
// tensor memory columns: G -> dk | dP -> dPsi2, O2 | O1 -> kD | xq | xk (bf16 A operands) | dq
constexpr int TM_A = 0, TM_B = 128, TM_O1 = 256, TM_XQ = 320, TM_XK = 352, TM_D = 384;
constexpr int TM_DK = TM_A, TM_DPSI = TM_B, TM_O2 = TM_B + 64, TM_KD = TM_O1;

struct __align__(1024) Smem {
  __nv_bfloat16 q[2 * TILE];   // Q -> qhat   [n half][row][64]
  __nv_bfloat16 k[2 * TILE];   // K -> khat
  __nv_bfloat16 pv[TILE];      // PsiV [row][p]
  __nv_bfloat16 dob[TILE];     // dO [row][p]
  __nv_bfloat16 std_[TILE];    // D_b, bf16 [n][p]
  __nv_bfloat16 gw[2 * TILE];  // G * W bf16, K-major [t][s] (s halves); staging of xq / xk before; Red after the kv products
  __nv_bfloat16 sth[TILE];     // h_b, bf16 [n][p]; from the masks on: first half of dP * W
  __nv_bfloat16 dpwb[TILE];    // second half of dP * W (follows sth)
  __nv_bfloat16 v[C * P], dy[C * P], z[C * P];  // raw [tok][p] (SW128)
  alignas(128) float angr[C * NQ];             // raw phases [tok][j] (TMA)
  alignas(16) float cs[C * NQP];  // cos table [tok][j]
  alignas(16) float sn[C * NQP];  // sin table [tok][j]
  float tab1[C * CP];             // W[t][s] = exp(l_t - l_s) f_s (s < t), gamma_t (s = t)
  alignas(16) __nv_bfloat16 qb[2][R * N], kb[2][R * N];
  alignas(16) float mv[2][R * PP], mo[2][R * PP], mz[2][R * PP];
  alignas(16) float ell[2][C], ff[2][C], gam[2][C], sc[2][4];  // sc: tot, D skip
  alignas(16) float ainb[2][NQ];                                // base phase of the sub-block's forward chunk
  alignas(16) float red[3][QC][ROWS];
  float dfi[4][C];             // intra part of the trapezoid factor gradient per token, per row quarter (q masks)
  float redk[QC][ROWS];        // kv side: inter part of the trapezoid factor gradient
  float ddk[THREADS / 32];     // kv side: dD partials per warp
  float xr[THREADS / 32 + 4];
  unsigned long long tA, tB, m1, m2a, m2b, mk1, mk2;
  uint32_t tmem_base;
};
static_assert(sizeof(Smem) + 1024 <= 232448, "shared memory");

struct Red {  // epilogue reduction buffers, overlaid on Smem::gw once the kv products are done with it
  float bias[4][R][N];         // per-warp-quarter partial bias gradients (dq / dk)
  float g1[4][R][P], g2[4][R][P];  // q: dmimo_o, dmimo_z partials; kv: dmimo_v partials
};
static_assert(sizeof(Red) <= TILE_B, "Red must fit in one tile");
struct Maps { CUtensorMap q, k, v, dy, z, ang, sth, std_; };

__device__ __forceinline__ int swz(int row, int col) {  // 128B swizzle, 64-element rows
  return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_wait(void* mbar, unsigned parity) {
  unsigned done = 0;
  while (!done) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(smem_u32(mbar)), "r"(parity) : "memory");
  }
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void* mbar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ uint64_t udesc(const void* p, uint32_t lbo, uint32_t sbo) {
  uint64_t d = (uint64_t)((smem_u32(p) & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint64_t kdesc(const void* p) { return udesc(p, 16, 1024); }      // K-major SW128
__device__ __forceinline__ uint64_t mdesc(const void* p) { return udesc(p, TILE_B, 1024); }  // MN-major SW128
__host__ __device__ constexpr uint32_t idesc(int M, int Nn, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) |
         ((uint32_t)(Nn >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void umma(uint32_t tmem_d, uint64_t a, uint64_t b, uint32_t id, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(tmem_d), "l"(a), "l"(b), "r"(id), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t b, uint32_t id, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
               :: "r"(tmem_d), "r"(tmem_a), "l"(b), "r"(id), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_commit(void* mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" :: "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]));
}
__device__ __forceinline__ void tmem_ld8(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
__device__ __forceinline__ void tmem_st8(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]));
}
template <int W> __device__ __forceinline__ void tmem_ldw(uint32_t taddr, float* v) {
  if constexpr (W == 32) tmem_ld32(taddr, v);
  else if constexpr (W == 16) tmem_ld16(taddr, v);
  else { static_assert(W == 8); tmem_ld8(taddr, v); }
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
template <int W> __device__ __forceinline__ void tmem_stw(uint32_t taddr, const uint32_t* r) {
  if constexpr (W == 16) tmem_st16(taddr, r);
  else { static_assert(W == 8); tmem_st8(taddr, r); }
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
union U8 { uint4 u; __nv_bfloat16 h[8]; __nv_bfloat162 h2[4]; };
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, void* mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma_store2d(const CUtensorMap* m, int c0, int c1, const void* src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
               :: "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(smem_u32(src)) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void tma3d(void* dst, const CUtensorMap* m, int c0, int c1, int c2, void* mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4z(void* dst, const void* src, bool ok) {  // zero-fill when !ok
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void red_add(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ float silu_f(float x) { return __fdividef(x, 1.f + __expf(-x)); }
// Reduce-scatter step over the lanes `lane ^ MASK`: v[0 .. NV/2) <- the pair sums of the half this lane keeps (the
// upper half v[NV/2 ..) if lane & MASK); half the shuffles of a full butterfly per step.
template <int NV, int MASK>
__device__ __forceinline__ void rs_step(float* v, int lane) {
  const bool up = lane & MASK;
#pragma unroll
  for (int i = 0; i < NV / 2; ++i) {
    const float send = up ? v[i] : v[i + NV / 2], keep = up ? v[i + NV / 2] : v[i];
    v[i] = keep + __shfl_xor_sync(0xffffffffu, send, MASK);
  }
}

constexpr int NACC = R * N + 2 * R * P + 4;  // per (CTA, head) accumulators: bias grad | mimo grads | dD
struct Args {
  const __nv_bfloat16 *qbias, *kbias;  // [H, R, N] bf16
  const float *mv, *mo, *mz, *Dskip, *ell, *ff, *gam, *tot;
  const int *sb_start, *sb_len;
  const float* ain;                    // [*, H, NQ] phase entering the forward chunk of a sub-block (ang is relative)
  const int* sb_chunk;                 // [NB] that chunk
  __nv_bfloat16 *sloc, *dloc;          // LOCAL: [NB, H, N, P] (bf16)
  __nv_bfloat16 *dq, *dk;              // summed over the heads  [S, R, N]
  __nv_bfloat16 *dz, *dv;              // [S, H, P]
  float* dth;                          // phase gradient (q + k parts)   [S, H, NQ]
  float *dlrow, *dgam, *dfac;          // [H, S]
  float *accq, *acck;                  // [gridDim, H, NACC], zero-initialised, owned by the CTA
  float* cross;                        // [NB, H] <h_{b+1}, D_b> = exp(tot_b) <h_b, D_b> + <Sloc_b, D_b>
  int S, H, NB;
  unsigned long long* prof;            // optional phase stamps (block 0, first 64 items)
};

#define STAMP(k) do { if (a.prof && blockIdx.x == 0 && tid == 0 && it < 64) a.prof[(64 + it) * 16 + (k)] = gtime(); } while (0)
// V2 FUSED kernel. Per (sub-block, head):
//   q first products  G = qhat khat^T (TM_A), dP = dO PsiV^T (TM_B), O1 = qhat h (TM_O1), dq state part
//                     (dO exp(l)) h^T (A = xq staged in TMEM, TM_D)
//   q masks           G * W, dP * W -> bf16 tiles in shared memory (gw, dpw = sth | dpwb, K-major [t][s]); the row
//                     sums of the q side and, by column reductions, the intra part of the trapezoid factor gradient
//                     (formerly the kv-side masks); the G / dP columns are free from here on
//   second products   q (thread 0): O2 = GW PsiV, dq += dPW khat;  kv (warp 12), right away: dPsi2 = GW^T dO,
//                     dk = xk D^T (A = xk in TMEM) + dPW^T qhat  (GW / dPW read as MN-major A operands): no GT / dPT
//                     products, no kv-side masks, and the kv products run under the q-side epilogues
//   kD = khat D       once the q outputs have read O1 (into its columns)
template <int MODE>
__global__ void __launch_bounds__(THREADS, 1) bwd_tc2_kernel(const __grid_constant__ Maps maps, const Args a) {
  extern __shared__ unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023));
  Red& RDQ = *reinterpret_cast<Red*>(S.gw);           // q-side reduction buffers (gw, first tile)
  Red& RDK = *reinterpret_cast<Red*>(S.gw + TILE);    // kv-side (second tile)
  __nv_bfloat16* const dpw = S.sth;  // dP * W: sth | dpwb (two adjacent tiles)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, H = a.H, NB = a.NB;
  if (blockIdx.x >= NB) return;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(smem_u32(&S.tmem_base)), "n"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  }
  if (tid == 0) {
    mbar_init(&S.tA, 1); mbar_init(&S.tB, 1);
    mbar_init(&S.m1, 4);  // G, O1, dP, dq state part
    mbar_init(&S.m2a, 1); mbar_init(&S.m2b, 1); mbar_init(&S.mk1, 1); mbar_init(&S.mk2, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = S.tmem_base;
  constexpr unsigned TXA = 6 * TILE_B, TXB = 3 * C * P * 2 + C * NQ * 4;
  // Loads of the next item: Q / K / state rows and V / dy / z / phases, all once the kv outputs are done with them
  // (single buffers; the next item's tables and qhat / khat build come first)
  auto issue_a = [&](int bb, int hh) {
    const int tt = a.sb_start[bb];
    mbar_arrive_expect_tx(&S.tA, TXA);
    tma2d(&S.q[0], &maps.q, 0, tt * R, &S.tA);
    tma2d(&S.q[TILE], &maps.q, 64, tt * R, &S.tA);
    tma2d(&S.k[0], &maps.k, 0, tt * R, &S.tA);
    tma2d(&S.k[TILE], &maps.k, 64, tt * R, &S.tA);
    tma2d(S.sth, &maps.sth, 0, (bb * H + hh) * N, &S.tA);
    tma2d(S.std_, &maps.std_, 0, (bb * H + hh) * N, &S.tA);
  };
  auto issue_b = [&](int bb, int hh) {
    const int tt = a.sb_start[bb];
    mbar_arrive_expect_tx(&S.tB, TXB);
    tma3d(S.v, &maps.v, 0, hh, tt, &S.tB);
    tma3d(S.dy, &maps.dy, 0, hh, tt, &S.tB);
    tma3d(S.z, &maps.z, 0, hh, tt, &S.tB);
    tma3d(S.angr, &maps.ang, 0, hh, tt, &S.tB);
  };
  auto prefetch_consts = [&](int bb, int hh, int cb) {  // head hh's constants, sub-block bb's token scalars
    for (int j = tid; j < R * N / 8; j += THREADS) {
      cp16(&S.qb[cb][8 * j], a.qbias + hh * R * N + 8 * j);
      cp16(&S.kb[cb][8 * j], a.kbias + hh * R * N + 8 * j);
    }
    for (int j = tid; j < R * P / 4; j += THREADS) {
      const int jj = (j >> 4) * PP + 4 * (j & 15);
      cp16(&S.mv[cb][jj], a.mv + hh * R * P + 4 * j);
      cp16(&S.mo[cb][jj], a.mo + hh * R * P + 4 * j);
      cp16(&S.mz[cb][jj], a.mz + hh * R * P + 4 * j);
    }
    if (tid < 3 * C) {
      const int w = tid / C, i = tid % C, tt = a.sb_start[bb];
      const bool ok = i < a.sb_len[bb];
      const float* src = w == 0 ? a.ell : w == 1 ? a.ff : a.gam;
      float* dst = w == 0 ? S.ell[cb] : w == 1 ? S.ff[cb] : S.gam[cb];
      cp4z(dst + i, src + (ok ? (long)hh * a.S + tt + i : 0), ok);
    } else if (tid == 3 * C) {
      cp4z(&S.sc[cb][0], a.tot + bb * H + hh, true);
    } else if (tid == 3 * C + 1) {
      cp4z(&S.sc[cb][1], a.Dskip + hh, true);
    } else if (tid >= 3 * C + 2 && tid < 3 * C + 2 + NQ / 4) {
      const int j = tid - (3 * C + 2);
      cp16(&S.ainb[cb][4 * j], a.ain + ((long)a.sb_chunk[bb] * H + hh) * NQ + 4 * j);
    }
    cp_commit();
  };
  if (tid == 0) { issue_a(blockIdx.x, 0); issue_b(blockIdx.x, 0); }
  prefetch_consts(blockIdx.x, 0, 0);
  // epilogue geometry: TMEM lanes 32 (warp % 4), column group warp / 4
  const int q4 = warp & 3, cq = warp >> 2, row = 32 * q4 + lane, tok = row >> 2, rank = row & 3;
  const uint32_t tl = tmem + ((uint32_t)(32 * q4) << 16);
  // producer geometry (rotation units): fixed rank / column piece per thread
  const int cr = tid & 3, rr = (tid >> 3) & 3, cA = 8 * cr, cB = 32 + 8 * cr;
  auto unit_row = [](int u) { return (u >> 5) * 8 + ((u >> 2) & 1) * 4 + ((u >> 3) & 3); };
  constexpr int JW = NQ / QC;   // rotated pairs per thread in the dq (dk) epilogue
  constexpr int OW = P / QC;    // O / PsiV columns per thread
  constexpr int GW = 128 / QC;  // mask columns per thread

  int it = 0;
  for (int b = blockIdx.x; b < NB; b += gridDim.x) {
    const int t0 = a.sb_start[b], nv = a.sb_len[b];
    const bool okr = tok < nv;
    const long grow = (long)t0 * R + row;  // global MIMO row
    float dxq[4][JW], dxk[4][JW];          // dq, dk of this thread's columns, summed over the heads
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
      for (int e = 0; e < JW; ++e) { dxq[x][e] = 0.f; dxk[x][e] = 0.f; }
    for (int h = 0; h < H; ++h, ++it) {
      float dthk0 = 0.f, dthk1 = 0.f;  // q part of this thread's phase gradient pairs
      int nb_ = b, nh_ = h + 1;
      if (nh_ == H) { nb_ = b + gridDim.x; nh_ = 0; }
      const bool has_next = nb_ < NB;
      const unsigned ph = it & 1;
      STAMP(0);
      // ---------------------------------------------------------------- per-head constants and token scalars
      const int cbf = it & 1;
      cp_wait_all();
      __syncthreads();  // this head's constants visible; the other buffer is free
      if (has_next) prefetch_consts(nb_, nh_, cbf ^ 1);
      const float* ellv = S.ell[cbf];
      const float* ffv = S.ff[cbf];
      const float* gamv = S.gam[cbf];
      const float* mvv = S.mv[cbf];
      const float* mov = S.mo[cbf];
      const float* mzv = S.mz[cbf];
      const float totv = S.sc[cbf][0], dskv = S.sc[cbf][1];
      mbar_wait(&S.tB, ph);
      STAMP(1);
      // ---------------------------------------------------------------- cos / sin table, mask table
      for (int idx = tid; idx < C * NQ; idx += THREADS) {
        float th = S.angr[idx] + S.ainb[cbf][idx & (NQ - 1)];
        th = th - 6.283185307179586f * floorf(th * 0.15915494309189535f);
        float sv, cv;
        __sincosf(th, &sv, &cv);
        S.cs[(idx >> 5) * NQP + (idx & 31)] = cv;
        S.sn[(idx >> 5) * NQP + (idx & 31)] = sv;
      }
      for (int idx = tid; idx < C * C; idx += THREADS) {  // W[t][s]
        const int t = idx >> 5, s = idx & 31;
        const bool valid = t < nv && s < nv;
        S.tab1[t * CP + s] = !valid ? 0.f : s < t ? __expf(fminf(ellv[t] - ellv[s], 0.f)) * ffv[s] : s == t ? gamv[t] : 0.f;
      }
      __syncthreads();
      mbar_wait(&S.tA, ph);
      STAMP(2);
      {  // exp(tot) <h_b, D_b> (the state tiles are reused below): 16 elements per thread, summed per warp
        static_assert(N * P == THREADS * 16, "16 state elements per thread");
        U8 h0, h1, d0, d1;
        h0.u = reinterpret_cast<const uint4*>(S.sth)[2 * tid];
        h1.u = reinterpret_cast<const uint4*>(S.sth)[2 * tid + 1];
        d0.u = reinterpret_cast<const uint4*>(S.std_)[2 * tid];
        d1.u = reinterpret_cast<const uint4*>(S.std_)[2 * tid + 1];
        float hd = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e)
          hd += __bfloat162float(h0.h[e]) * __bfloat162float(d0.h[e]) + __bfloat162float(h1.h[e]) * __bfloat162float(d1.h[e]);
#pragma unroll
        for (int m = 1; m <= 16; m <<= 1) hd += __shfl_xor_sync(0xffffffffu, hd, m);
        if (lane == 0) S.xr[warp] = __expf(totv) * hd;
      }
      // ---------------------------------------------------------------- operands
      {  // qhat, khat in place (bias in bf16, rotation of the (n, n + 64) pairs, n < 32)
        U8 bq[4], bk[4];
        bq[0].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cA]);
        bq[1].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cA + 64]);
        bq[2].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cB]);
        bq[3].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cB + 64]);
        bk[0].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cA]);
        bk[1].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cA + 64]);
        bk[2].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cB]);
        bk[3].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cB + 64]);
#pragma unroll 1
        for (int u = tid; u < ROWS * 4; u += THREADS) {
          const int r_ = unit_row(u), tk = r_ >> 2;
          const bool ok = tk < nv;
          const int oA = swz(r_, cA), oB = swz(r_, cB);
          U8 qv[4], kv[4];
          qv[0].u = *reinterpret_cast<const uint4*>(&S.q[oA]);
          qv[1].u = *reinterpret_cast<const uint4*>(&S.q[TILE + oA]);
          qv[2].u = *reinterpret_cast<const uint4*>(&S.q[oB]);
          qv[3].u = *reinterpret_cast<const uint4*>(&S.q[TILE + oB]);
          kv[0].u = *reinterpret_cast<const uint4*>(&S.k[oA]);
          kv[1].u = *reinterpret_cast<const uint4*>(&S.k[TILE + oA]);
          kv[2].u = *reinterpret_cast<const uint4*>(&S.k[oB]);
          kv[3].u = *reinterpret_cast<const uint4*>(&S.k[TILE + oB]);
#pragma unroll
          for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              qv[x].h2[j] = __hadd2(qv[x].h2[j], bq[x].h2[j]);
              kv[x].h2[j] = __hadd2(kv[x].h2[j], bk[x].h2[j]);
            }
          const float* cp = &S.cs[tk * NQP + cA];
          const float* sp = &S.sn[tk * NQP + cA];
#pragma unroll
          for (int j = 0; j < 4; ++j) {  // rotations in packed fp32 (column pairs)
            const float2 c2 = make_float2(cp[2 * j], cp[2 * j + 1]), s2 = make_float2(sp[2 * j], sp[2 * j + 1]);
            const float2 ns2 = make_float2(-s2.x, -s2.y);
            const float2 x0 = __bfloat1622float2(qv[0].h2[j]), x1 = __bfloat1622float2(qv[1].h2[j]);
            const float2 y0 = __bfloat1622float2(kv[0].h2[j]), y1 = __bfloat1622float2(kv[1].h2[j]);
            qv[0].h2[j] = __float22bfloat162_rn(__ffma2_rn(ns2, x1, __fmul2_rn(c2, x0)));
            qv[1].h2[j] = __float22bfloat162_rn(__ffma2_rn(c2, x1, __fmul2_rn(s2, x0)));
            kv[0].h2[j] = __float22bfloat162_rn(__ffma2_rn(ns2, y1, __fmul2_rn(c2, y0)));
            kv[1].h2[j] = __float22bfloat162_rn(__ffma2_rn(c2, y1, __fmul2_rn(s2, y0)));
          }
          if (!ok) {
#pragma unroll
            for (int x = 0; x < 4; ++x) { qv[x].u = make_uint4(0, 0, 0, 0); kv[x].u = make_uint4(0, 0, 0, 0); }
          }
          *reinterpret_cast<uint4*>(&S.q[oA]) = qv[0].u;
          *reinterpret_cast<uint4*>(&S.q[TILE + oA]) = qv[1].u;
          *reinterpret_cast<uint4*>(&S.q[oB]) = qv[2].u;
          *reinterpret_cast<uint4*>(&S.q[TILE + oB]) = qv[3].u;
          *reinterpret_cast<uint4*>(&S.k[oA]) = kv[0].u;
          *reinterpret_cast<uint4*>(&S.k[TILE + oA]) = kv[1].u;
          *reinterpret_cast<uint4*>(&S.k[oB]) = kv[2].u;
          *reinterpret_cast<uint4*>(&S.k[TILE + oB]) = kv[3].u;
        }
      }
      STAMP(3);
      fence_async();
      tc_fence_before();
      __syncthreads();
      {  // G = qhat khat^T, O1 = qhat h run while PsiV / dO are built
        constexpr uint32_t ID = idesc(128, 128, 0, 0), ID_O = idesc(128, 64, 0, 1);
        if (tid == 0) {
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)
            umma(tmem + TM_A, kdesc(&S.q[(kk >> 2) * TILE + (kk & 3) * 16]), kdesc(&S.k[(kk >> 2) * TILE + (kk & 3) * 16]), ID, kk > 0);
          umma_commit(&S.m1);
        } else if (tid == 256) {
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)
            umma(tmem + TM_O1, kdesc(&S.q[(kk >> 2) * TILE + (kk & 3) * 16]), mdesc(&S.sth[kk * 16 * 64]), ID_O, kk > 0);
          umma_commit(&S.m1);
        }
      }
      // PsiV, dO -> pv / dob, in the epilogue geometry (row, 16 columns): dO exp(l) and PsiV exp(l_end - l) f go
      // straight to this thread's tensor-memory lanes (bf16 pairs, A operands of the state parts of dq / dk)
      {
        const int p0 = OW * cq;
        const float eq = okr ? __expf(ellv[tok]) : 0.f;
        const float ek = okr ? __expf(totv - ellv[tok]) * ffv[tok] : 0.f;
        const float2 ek2 = make_float2(ek, ek), eq2 = make_float2(eq, eq);
        uint32_t xq8[8], xk8[8];
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int pc = p0 + 8 * hf;
          U8 vv, dd, zz, o1, o2;
          vv.u = *reinterpret_cast<const uint4*>(&S.v[swz(tok, pc)]);
          dd.u = *reinterpret_cast<const uint4*>(&S.dy[swz(tok, pc)]);
          zz.u = *reinterpret_cast<const uint4*>(&S.z[swz(tok, pc)]);
          float cmv[8], cmo[8], cmz[8];
          *reinterpret_cast<float4*>(cmv) = *reinterpret_cast<const float4*>(&mvv[rank * PP + pc]);
          *reinterpret_cast<float4*>(cmv + 4) = *reinterpret_cast<const float4*>(&mvv[rank * PP + pc + 4]);
          *reinterpret_cast<float4*>(cmo) = *reinterpret_cast<const float4*>(&mov[rank * PP + pc]);
          *reinterpret_cast<float4*>(cmo + 4) = *reinterpret_cast<const float4*>(&mov[rank * PP + pc + 4]);
          *reinterpret_cast<float4*>(cmz) = *reinterpret_cast<const float4*>(&mzv[rank * PP + pc]);
          *reinterpret_cast<float4*>(cmz + 4) = *reinterpret_cast<const float4*>(&mzv[rank * PP + pc + 4]);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {  // column pairs in packed fp32
            const int e2 = e >> 1;
            const float2 cv = make_float2(cmv[e], cmv[e + 1]), cz = make_float2(cmz[e], cmz[e + 1]);
            const float2 co = make_float2(cmo[e], cmo[e + 1]);
            float2 ps = __fmul2_rn(__bfloat1622float2(vv.h2[e2]), cv);
            const float2 zm = __fmul2_rn(__bfloat1622float2(zz.h2[e2]), cz);
            float2 dO = __fmul2_rn(__fmul2_rn(__bfloat1622float2(dd.h2[e2]), co), make_float2(silu_f(zm.x), silu_f(zm.y)));
            if (!okr) { ps = make_float2(0.f, 0.f); dO = make_float2(0.f, 0.f); }
            o1.h2[e2] = __float22bfloat162_rn(ps);
            o2.h2[e2] = __float22bfloat162_rn(dO);
            const __nv_bfloat162 xq2 = __float22bfloat162_rn(__fmul2_rn(dO, eq2));
            const __nv_bfloat162 xk2 = __float22bfloat162_rn(__fmul2_rn(ps, ek2));
            xq8[4 * hf + e2] = *reinterpret_cast<const uint32_t*>(&xq2);
            xk8[4 * hf + e2] = *reinterpret_cast<const uint32_t*>(&xk2);
          }
          *reinterpret_cast<uint4*>(&S.pv[swz(row, pc)]) = o1.u;
          *reinterpret_cast<uint4*>(&S.dob[swz(row, pc)]) = o2.u;
        }
        tmem_st8(tl + TM_XQ + (OW / 2) * cq, xq8);
        tmem_st8(tl + TM_XK + (OW / 2) * cq, xk8);
        tmem_wait_st();
      }
      STAMP(4);
      fence_async();
      tc_fence_before();
      __syncthreads();
      {  // dP, and the state part of dq (A = xq from tensor memory)
        constexpr uint32_t ID = idesc(128, 128, 0, 0);
        if (tid == 128) {
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) umma(tmem + TM_B, kdesc(&S.dob[kk * 16]), kdesc(&S.pv[kk * 16]), ID, kk > 0);
          umma_commit(&S.m1);
        } else if (tid == 384) {
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) umma_ts(tmem + TM_D, tmem + TM_XQ + 8 * kk, kdesc(&S.sth[kk * 16]), ID, kk > 0);
          umma_commit(&S.m1);
        }
      }
      mbar_wait(&S.m1, ph);
      tc_fence_after();
      STAMP(5);
      // ---------------------------------------------------------------- q masks -> bf16 G * W, dP * W tiles
      {
        float acc1 = 0.f, acc2 = 0.f;
        const int cb = GW * cq;
        float g[GW], d[GW];
        tmem_ldw<GW>(tl + TM_A + cb, g);
        tmem_ldw<GW>(tl + TM_B + cb, d);
        tmem_wait_ld();
        STAMP(15);
        uint32_t pg[GW / 2], pd[GW / 2];
        float xf[GW / 4];  // per column token: sum over its ranks of A[t][s] times dec[s][t] (trapezoid factor, intra)
        const float lt = ellv[tok];
#pragma unroll
        for (int e = 0; e < GW; e += 4) {
          const int ts = (cb + e) >> 2;  // the four columns e .. e + 3 are the ranks of token ts
          const float w = S.tab1[tok * CP + ts];  // W[t = row][s = col]
          const float2 g0 = make_float2(g[e], g[e + 1]), d0 = make_float2(d[e], d[e + 1]);
          const float2 g1 = make_float2(g[e + 2], g[e + 3]), d1 = make_float2(d[e + 2], d[e + 3]);
          const float2 A0 = __fmul2_rn(g0, d0), A1 = __fmul2_rn(g1, d1);
          const float sA0 = A0.x + A0.y, sA1 = A1.x + A1.y;
          if (ts < tok) { acc1 += sA0 * w; acc1 += sA1 * w; }
          else if (ts == tok) { acc2 += sA0; acc2 += sA1; }
          const float dec = (okr && ts < nv && ts < tok) ? __expf(fminf(lt - ellv[ts], 0.f)) : 0.f;
          xf[e >> 2] = (sA0 + sA1) * dec;
          const float2 ww = make_float2(w, w);
          const float2 gw0 = __fmul2_rn(g0, ww), dw0 = __fmul2_rn(d0, ww), gw1 = __fmul2_rn(g1, ww), dw1 = __fmul2_rn(d1, ww);
          pg[e >> 1] = pack_bf16(gw0.x, gw0.y); pg[(e >> 1) + 1] = pack_bf16(gw1.x, gw1.y);
          pd[e >> 1] = pack_bf16(dw0.x, dw0.y); pd[(e >> 1) + 1] = pack_bf16(dw1.x, dw1.y);
        }
        STAMP(9);
#pragma unroll
        for (int m = 0; m < GW / 8; ++m) {  // 8-column chunks of the K-major [t][s] tiles (two 64-column halves)
          const int cc = (cb >> 3) + m;
          const int off = (cc >> 3) * TILE + swz(row, (cc & 7) * 8);
          *reinterpret_cast<uint4*>(&S.gw[off]) = make_uint4(pg[4 * m], pg[4 * m + 1], pg[4 * m + 2], pg[4 * m + 3]);
          *reinterpret_cast<uint4*>(&dpw[off]) = make_uint4(pd[4 * m], pd[4 * m + 1], pd[4 * m + 2], pd[4 * m + 3]);
        }
        STAMP(10);
        // column sums over the rows t: reduce-scatter over the lanes, then the lanes of equal (lane & 7) sum the rest
        static_assert(GW / 4 == 8, "reduce-scatter layout");
        rs_step<8, 1>(xf, lane);
        rs_step<4, 2>(xf, lane);
        rs_step<2, 4>(xf, lane);
        xf[0] += __shfl_xor_sync(0xffffffffu, xf[0], 8);
        xf[0] += __shfl_xor_sync(0xffffffffu, xf[0], 16);
        if (lane < 8) S.dfi[q4][(cb >> 2) + 4 * (lane & 1) + 2 * ((lane >> 1) & 1) + ((lane >> 2) & 1)] = xf[0];
        S.red[0][cq][row] = acc1;
        S.red[1][cq][row] = acc2;
      }
      STAMP(6);
      fence_async();
      tc_fence_before();
      __syncthreads();
      // ---------------------------------------------------------------- second products
      if (tid == 0) {  // q: O2 first (the outputs wait for it), then the dq chain
        constexpr uint32_t ID_O = idesc(128, 64, 0, 1);
        tc_fence_after();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)  // O2 = (G W) PsiV
          umma(tmem + TM_O2, kdesc(&S.gw[(kk >> 2) * TILE + (kk & 3) * 16]), mdesc(&S.pv[kk * 16 * 64]), ID_O, kk > 0);
        umma_commit(&S.m2a);
        constexpr uint32_t ID_X = idesc(128, 128, 0, 1);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)  // dq += (dP W) khat, queued behind O2
          umma(tmem + TM_D, kdesc(&dpw[(kk >> 2) * TILE + (kk & 3) * 16]), mdesc(&S.k[kk * 16 * 64]), ID_X, 1);
        umma_commit(&S.m2b);
      }
      mbar_wait(&S.m2a, ph);
      tc_fence_after();
      STAMP(7);
      if (tid == 384) {  // kv: dPsi2 = (G W)^T dO, dk = xk D^T + (dP W)^T qhat
        constexpr uint32_t ID_T = idesc(128, 64, 1, 1), ID_TT = idesc(128, 128, 1, 1), ID_I = idesc(128, 128, 0, 0);
        tc_fence_after();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
          umma(tmem + TM_DPSI, mdesc(&S.gw[kk * 16 * 64]), mdesc(&S.dob[kk * 16 * 64]), ID_T, kk > 0);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) umma_ts(tmem + TM_DK, tmem + TM_XK + 8 * kk, kdesc(&S.std_[kk * 16]), ID_I, kk > 0);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) umma(tmem + TM_DK, mdesc(&dpw[kk * 16 * 64]), mdesc(&S.q[kk * 16 * 64]), ID_TT, 1);
        umma_commit(&S.mk1);
      }
      // ---------------------------------------------------------------- the two sides' epilogues
      auto side = [&](auto side_tag, float (&dxa)[4][JW]) {
        constexpr int SD = decltype(side_tag)::value;
        constexpr bool QS = SD == SIDE_Q;
        const __nv_bfloat16* X = QS ? S.q : S.k;  // rows of this side: qhat (q side) / khat (kv side)
        const uint32_t TO1 = QS ? TM_O1 : TM_KD, TO2 = QS ? TM_O2 : TM_DPSI, TD = QS ? TM_D : TM_DK;
        if (!QS) {
          mbar_wait(&S.mk2, ph);  // kD (mk1, the rest of the kv products, was waited on the q side)
          tc_fence_after();
          STAMP(11);
        }
        float sc3 = 0.f;
        float g1[OW], g2[OW], dz[OW];
        float pdl = 0.f;
        {  // O1 / O2 columns p in [OW cq, OW cq + OW): q: o -> gate gradients; kv: dPsiV -> dv, dmimo_v, dD
          float o1[OW], o2[OW];
          tmem_ldw<OW>(tl + TO1 + OW * cq, o1);
          tmem_ldw<OW>(tl + TO2 + OW * cq, o2);
          tmem_wait_ld();
          const float eq = okr ? __expf(ellv[tok]) : 0.f;
          const float ek0 = okr ? __expf(totv - ellv[tok]) : 0.f, ek = ek0 * ffv[tok];
#pragma unroll
          for (int ch = 0; ch < OW / 8; ++ch) {
            const int p0 = OW * cq + 8 * ch;
            U8 pv, dd, ry, rz, rv;
            pv.u = *reinterpret_cast<const uint4*>(&S.pv[swz(row, p0)]);
            dd.u = *reinterpret_cast<const uint4*>(&S.dob[swz(row, p0)]);
            float cmo[8], cmz[8], cmv[8];  // q side: mimo_o, mimo_z; kv side: mimo_v
            if (QS) {
              *reinterpret_cast<float4*>(cmo) = *reinterpret_cast<const float4*>(&mov[rank * PP + p0]);
              *reinterpret_cast<float4*>(cmo + 4) = *reinterpret_cast<const float4*>(&mov[rank * PP + p0 + 4]);
              *reinterpret_cast<float4*>(cmz) = *reinterpret_cast<const float4*>(&mzv[rank * PP + p0]);
              *reinterpret_cast<float4*>(cmz + 4) = *reinterpret_cast<const float4*>(&mzv[rank * PP + p0 + 4]);
            } else {
              *reinterpret_cast<float4*>(cmv) = *reinterpret_cast<const float4*>(&mvv[rank * PP + p0]);
              *reinterpret_cast<float4*>(cmv + 4) = *reinterpret_cast<const float4*>(&mvv[rank * PP + p0 + 4]);
            }
            if (QS) {
              ry.u = *reinterpret_cast<const uint4*>(&S.dy[swz(tok, p0)]);
              rz.u = *reinterpret_cast<const uint4*>(&S.z[swz(tok, p0)]);
            } else {
              rv.u = *reinterpret_cast<const uint4*>(&S.v[swz(tok, p0)]);
            }
#pragma unroll
            for (int e = 0; e < 8; e += 2) {  // column pairs in packed fp32 (FMUL2 / FFMA2)
              const int i = 8 * ch + e;
              const float2 ps = __bfloat1622float2(pv.h2[e >> 1]), dO = __bfloat1622float2(dd.h2[e >> 1]);
              const float2 o1v = make_float2(o1[i], o1[i + 1]), o2v = make_float2(o2[i], o2[i + 1]);
              const float2 dsk2 = make_float2(dskv, dskv);
              if (QS) {
                const float2 eq2 = make_float2(eq, eq);
                const float2 eo1 = __fmul2_rn(eq2, o1v);
                const float2 o = __ffma2_rn(dsk2, ps, __fadd2_rn(eo1, o2v));   // pre-gate output (recomputed)
                const float2 sr = __fmul2_rn(eo1, dO);                          // inter part of the decay row sum
                sc3 += sr.x + sr.y;
                const float2 y = okr ? __bfloat1622float2(ry.h2[e >> 1]) : make_float2(0.f, 0.f);
                const float2 zv = __bfloat1622float2(rz.h2[e >> 1]);
                const float2 cz = make_float2(cmz[e], cmz[e + 1]), co = make_float2(cmo[e], cmo[e + 1]);
                const float2 zz = __fmul2_rn(zv, cz);
                const float2 sg = make_float2(1.f / (1.f + __expf(-zz.x)), 1.f / (1.f + __expf(-zz.y)));
                const float2 one = make_float2(1.f, 1.f);
                const float2 inner = __ffma2_rn(zz, __fadd2_rn(one, make_float2(-sg.x, -sg.y)), one);  // 1 + zz (1 - sg)
                const float2 yos = __fmul2_rn(__fmul2_rn(y, o), sg);
                const float2 dzz = __fmul2_rn(__fmul2_rn(yos, co), inner);
                const float2 gg1 = __fmul2_rn(yos, zz);   // dmimo_o
                const float2 gg2 = __fmul2_rn(dzz, zv);   // dmimo_z
                const float2 ddz = __fmul2_rn(dzz, cz);   // dz (summed over the ranks below)
                g1[i] = gg1.x; g1[i + 1] = gg1.y;
                g2[i] = gg2.x; g2[i + 1] = gg2.y;
                dz[i] = ddz.x; dz[i + 1] = ddz.y;
              } else {
                const float2 ek2 = make_float2(ek, ek), ek02 = make_float2(ek0, ek0);
                float2 dps = __ffma2_rn(dsk2, dO, __ffma2_rn(ek2, o1v, o2v));  // dPsiV
                if (!okr) dps = make_float2(0.f, 0.f);
                const float2 sr = __fmul2_rn(__fmul2_rn(ek02, o1v), ps);       // kD . PsiV (factor gradient)
                sc3 += sr.x + sr.y;
                const float2 vv = okr ? __bfloat1622float2(rv.h2[e >> 1]) : make_float2(0.f, 0.f);
                const float2 gg1 = __fmul2_rn(dps, vv);   // dmimo_v
                const float2 ddv = __fmul2_rn(dps, make_float2(cmv[e], cmv[e + 1]));  // dv (summed over the ranks below)
                const float2 pd = __fmul2_rn(dO, ps);     // dD
                g1[i] = gg1.x; g1[i + 1] = gg1.y;
                g2[i] = 0.f; g2[i + 1] = 0.f;
                dz[i] = ddv.x; dz[i + 1] = ddv.y;
                pdl += pd.x + pd.y;
              }
            }
          }
        }
        // per-token sums over the ranks (lanes 4m .. 4m + 3), reduce-scattered: dz / dv columns
        // OW cq + 8 (rank & 1) + 4 (rank >> 1) + [0, 4) on every lane of the token
        static_assert(OW == 16, "reduce-scatter layout");
        rs_step<16, 1>(dz, lane);
        rs_step<8, 2>(dz, lane);
        if (okr) {
          uint2* dst = reinterpret_cast<uint2*>((QS ? a.dz : a.dv) + ((long)(t0 + tok) * H + h) * P + OW * cq
                                                + 8 * (rank & 1) + 4 * (rank >> 1));
          *dst = make_uint2(pack_bf16(dz[0], dz[1]), pack_bf16(dz[2], dz[3]));
        }
        rs_step<16, 4>(g1, lane); rs_step<8, 8>(g1, lane); rs_step<4, 16>(g1, lane);
        if (QS) { rs_step<16, 4>(g2, lane); rs_step<8, 8>(g2, lane); rs_step<4, 16>(g2, lane); }
        const int gcol = OW * cq + 8 * ((lane >> 2) & 1) + 4 * ((lane >> 3) & 1) + 2 * ((lane >> 4) & 1);
        if (!QS) {
#pragma unroll
          for (int m = 1; m <= 16; m <<= 1) pdl += __shfl_xor_sync(0xffffffffu, pdl, m);
        }
        STAMP(QS ? 8 : 12);
        float xh0[JW], xh1[JW];  // qhat (khat) values of this thread's rotated pairs, for the phase gradient
        {
          static_assert(JW == 8, "one 16-byte piece");
          U8 u0, u1;
          u0.u = *reinterpret_cast<const uint4*>(&X[swz(row, JW * cq)]);
          u1.u = *reinterpret_cast<const uint4*>(&X[TILE + swz(row, JW * cq)]);
#pragma unroll
          for (int e = 0; e < JW; ++e) { xh0[e] = __bfloat162float(u0.h[e]); xh1[e] = __bfloat162float(u1.h[e]); }
        }
        if (QS) {
          mbar_wait(&S.m2b, ph);
          tc_fence_after();
          tc_fence_before();
          __syncthreads();  // every thread has read O1: kD takes its columns
          if (tid == 256) {
            constexpr uint32_t ID_O = idesc(128, 64, 0, 1);
            tc_fence_after();
#pragma unroll
            for (int kk = 0; kk < 8; ++kk)  // kD = khat D
              umma(tmem + TM_KD, kdesc(&S.k[(kk >> 2) * TILE + (kk & 3) * 16]), mdesc(&S.std_[kk * 16 * 64]), ID_O, kk > 0);
            umma_commit(&S.mk2);
          }
          mbar_wait(&S.mk1, ph);  // the kv products are done with gw / dpw: the reduction buffers overlay gw
          tc_fence_after();
        } else {
          tc_fence_before();
          __syncthreads();  // every thread is done with V / dy / z and the Q / K / state tiles
          if (tid == 0 && has_next) { issue_a(nb_, nh_); issue_b(nb_, nh_); }
          tc_fence_after();
        }
        Red& RD = QS ? RDQ : RDK;
        *reinterpret_cast<float2*>(&RD.g1[q4][rank][gcol]) = make_float2(g1[0], g1[1]);
        if (QS) *reinterpret_cast<float2*>(&RD.g2[q4][rank][gcol]) = make_float2(g2[0], g2[1]);
        if (!QS && lane == 0) S.ddk[warp] = pdl;  // dD partial of this warp
        {  // dq (dk): rotated pairs j in [JW cq, JW cq + JW) and the plain columns 32 + JW cq.., 96 + JW cq..
          float d0[JW], d1[JW], p0v[JW], p1v[JW];
          tmem_ldw<JW>(tl + TD + JW * cq, d0);
          tmem_ldw<JW>(tl + TD + 64 + JW * cq, d1);
          tmem_ldw<JW>(tl + TD + 32 + JW * cq, p0v);
          tmem_ldw<JW>(tl + TD + 96 + JW * cq, p1v);
          tmem_wait_ld();
          float dth[JW], bx[4][JW];
#pragma unroll
          for (int e = 0; e < JW; e += 2) {  // column pairs in packed fp32
            const int j = JW * cq + e;
            const float2 c = make_float2(S.cs[tok * NQP + j], S.cs[tok * NQP + j + 1]);
            const float2 sn = make_float2(S.sn[tok * NQP + j], S.sn[tok * NQP + j + 1]);
            const float2 a0 = make_float2(d0[e], d0[e + 1]), a1 = make_float2(d1[e], d1[e + 1]);
            const float2 na0 = make_float2(-a0.x, -a0.y);
            float2 r0 = __ffma2_rn(sn, a1, __fmul2_rn(c, a0));   // c a0 + s a1
            float2 r1 = __ffma2_rn(sn, na0, __fmul2_rn(c, a1));  // c a1 - s a0
            float2 r2 = make_float2(p0v[e], p0v[e + 1]), r3 = make_float2(p1v[e], p1v[e + 1]);
            if (!okr) r0 = r1 = r2 = r3 = make_float2(0.f, 0.f);
            const float2 th = __ffma2_rn(na0, make_float2(xh1[e], xh1[e + 1]), __fmul2_rn(a1, make_float2(xh0[e], xh0[e + 1])));
            bx[0][e] = r0.x; bx[0][e + 1] = r0.y; bx[1][e] = r1.x; bx[1][e + 1] = r1.y;
            bx[2][e] = r2.x; bx[2][e + 1] = r2.y; bx[3][e] = r3.x; bx[3][e + 1] = r3.y;
            dth[e] = th.x; dth[e + 1] = th.y;
#pragma unroll
            for (int x = 0; x < 4; ++x) {
              const float2 acc = __fadd2_rn(make_float2(dxa[x][e], dxa[x][e + 1]), make_float2(bx[x][e], bx[x][e + 1]));
              dxa[x][e] = acc.x; dxa[x][e + 1] = acc.y;
            }
          }
          static_assert(JW == 8, "reduce-scatter layout");
          rs_step<8, 1>(dth, lane);  // over the ranks: pairs JW cq + 4 (rank & 1) + 2 (rank >> 1) + [0, 2)
          rs_step<4, 2>(dth, lane);
          if (QS) {
            dthk0 = dth[0]; dthk1 = dth[1];
          } else if (okr) {
            float* dst = a.dth + ((long)(t0 + tok) * H + h) * NQ + JW * cq + 4 * (rank & 1) + 2 * (rank >> 1);
            *reinterpret_cast<float2*>(dst) = make_float2(dth[0] + dthk0, dth[1] + dthk1);
          }
          float* bf = &bx[0][0];
          rs_step<32, 4>(bf, lane); rs_step<16, 8>(bf, lane); rs_step<8, 16>(bf, lane);
          {
            const int f0 = 16 * ((lane >> 2) & 1) + 8 * ((lane >> 3) & 1) + 4 * ((lane >> 4) & 1);
            const int x = f0 >> 3, j = JW * cq + (f0 & 7);
            const int col = (x == 0 ? 0 : x == 1 ? 64 : x == 2 ? 32 : 96) + j;
            *reinterpret_cast<float4*>(&RD.bias[q4][rank][col]) = make_float4(bf[0], bf[1], bf[2], bf[3]);
          }
        }
        if (QS) {  // the q side's per-token sums and accumulators run with the kv side's, at the end of the item
          S.red[2][cq][row] = sc3;
          return;
        }
        S.redk[cq][row] = sc3;
        STAMP(13);
        __syncthreads();
        if (cq == 0) {  // per-token sums (sum over the column groups, then over the ranks), both sides
          float v1 = 0.f, v2 = 0.f, vk = 0.f;
#pragma unroll
          for (int g = 0; g < QC; ++g) {
            v1 += S.red[0][g][row] + S.red[2][g][row];  // q: dl row (intra + inter)
            v2 += S.red[1][g][row];                     // q: diagonal (gamma)
            vk += S.redk[g][row];                       // kv: inter part of dfactor
          }
          const float xc0 = vk;
#pragma unroll
          for (int m = 1; m <= 2; m <<= 1) {
            v1 += __shfl_xor_sync(0xffffffffu, v1, m);
            v2 += __shfl_xor_sync(0xffffffffu, v2, m);
            vk += __shfl_xor_sync(0xffffffffu, vk, m);
          }
          if (okr && rank == 0) {
            const long o = (long)h * a.S + t0 + tok;
            a.dlrow[o] = v1;
            a.dgam[o] = v2;
            a.dfac[o] = vk + ((S.dfi[0][tok] + S.dfi[1][tok]) + S.dfi[2][tok]) + S.dfi[3][tok];
          }
          // <Sloc_b, D_b> = sum_s f_s exp(l_end - l_s) kD_s . PsiV_s (the inter part of dfactor, times f)
          float xc = okr ? xc0 * ffv[tok] : 0.f;
#pragma unroll
          for (int m = 1; m <= 16; m <<= 1) xc += __shfl_xor_sync(0xffffffffu, xc, m);
          if (lane == 0) S.xr[THREADS / 32 + q4] = xc;
        }
        {  // per (CTA, head) accumulators: fixed summation order, one owning thread per address (deterministic)
          float* accq = a.accq + ((long)blockIdx.x * H + h) * NACC;
          float* acck = a.acck + ((long)blockIdx.x * H + h) * NACC;
          for (int j = tid; j < R * N; j += THREADS) {
            const int r_ = j >> 7, n = j & 127;
            red_add(accq + j, ((RDQ.bias[0][r_][n] + RDQ.bias[1][r_][n]) + RDQ.bias[2][r_][n]) + RDQ.bias[3][r_][n]);
            red_add(acck + j, ((RDK.bias[0][r_][n] + RDK.bias[1][r_][n]) + RDK.bias[2][r_][n]) + RDK.bias[3][r_][n]);
          }
          for (int j = tid; j < R * P; j += THREADS) {
            const int r_ = j >> 6, p = j & 63;
            red_add(accq + R * N + j, ((RDQ.g1[0][r_][p] + RDQ.g1[1][r_][p]) + RDQ.g1[2][r_][p]) + RDQ.g1[3][r_][p]);
            red_add(accq + R * N + R * P + j, ((RDQ.g2[0][r_][p] + RDQ.g2[1][r_][p]) + RDQ.g2[2][r_][p]) + RDQ.g2[3][r_][p]);
            red_add(acck + R * N + j, ((RDK.g1[0][r_][p] + RDK.g1[1][r_][p]) + RDK.g1[2][r_][p]) + RDK.g1[3][r_][p]);
          }
          if (tid == 0) {
            float s = 0.f;
            for (int g = 0; g < QC; ++g)
              for (int w = 0; w < 4; ++w) s += S.ddk[4 * g + w];
            red_add(acck + R * N + 2 * R * P, s);
          }
        }
        tc_fence_before();
        __syncthreads();
        STAMP(14);
        if (tid == 0) {  // fixed order (deterministic)
          float c = 0.f;
#pragma unroll
          for (int i = 0; i < THREADS / 32 + 4; ++i) c += S.xr[i];
          a.cross[(long)b * H + h] = c;
        }
      };
      side(std::integral_constant<int, SIDE_Q>{}, dxq);
      side(std::integral_constant<int, SIDE_KV>{}, dxk);
    }
    if (okr) {  // dq, dk summed over the heads
#pragma unroll
      for (int sd = 0; sd < 2; ++sd) {
        __nv_bfloat16* dst = (sd == 0 ? a.dq : a.dk) + grow * N + JW * cq;
        float (&dxa)[4][JW] = sd == 0 ? dxq : dxk;
#pragma unroll
        for (int x = 0; x < JW / 8; ++x) {
          uint4 u[4];
#pragma unroll
          for (int g = 0; g < 4; ++g)
            u[g] = make_uint4(pack_bf16(dxa[g][8 * x], dxa[g][8 * x + 1]), pack_bf16(dxa[g][8 * x + 2], dxa[g][8 * x + 3]),
                              pack_bf16(dxa[g][8 * x + 4], dxa[g][8 * x + 5]), pack_bf16(dxa[g][8 * x + 6], dxa[g][8 * x + 7]));
          *reinterpret_cast<uint4*>(dst + 8 * x) = u[0];
          *reinterpret_cast<uint4*>(dst + 64 + 8 * x) = u[1];
          *reinterpret_cast<uint4*>(dst + 32 + 8 * x) = u[2];
          *reinterpret_cast<uint4*>(dst + 96 + 8 * x) = u[3];
        }
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tmem), "n"(512));
  }
}

}  // namespace

// ---------------------------------------------------------------- host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres;
    void* p = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qres);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}
static CUtensorMap make_map(const void* base, CUtensorMapDataType dt, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                            const cuuint32_t* box, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  const cuuint32_t estr[3] = {1, 1, 1};
  const CUresult r = get_encode_fn()(&m, dt, rank, const_cast<void*>(base), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
static CUtensorMap state_map(const torch::Tensor& st) {
  TORCH_CHECK(st.is_contiguous() && st.scalar_type() == torch::kBFloat16 && st.size(-1) == P && st.size(-2) == N);
  const cuuint64_t d[2] = {(cuuint64_t)P, (cuuint64_t)(st.numel() / P)}, s[1] = {(cuuint64_t)P * 2};
  const cuuint32_t box[2] = {64, ROWS};
  return make_map(st.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, s, box, CU_TENSOR_MAP_SWIZZLE_128B);
}

static Maps build_maps(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor DY, torch::Tensor Z,
                       torch::Tensor ang, const torch::Tensor* hs, const torch::Tensor* ds) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && Q.size(-1) == N && Q.numel() == (long)T * R * N);
  for (auto* x : {&V, &DY, &Z})
    TORCH_CHECK(x->size(0) == T && x->size(1) == H && x->size(2) == P && x->stride(2) == 1 && x->stride(1) == P &&
                x->stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x->data_ptr()) & 15) == 0);
  TORCH_CHECK(ang.is_contiguous() && ang.size(-1) == NQ && ang.numel() == (long)T * H * NQ);
  Maps M;
  {
    const cuuint64_t d[2] = {(cuuint64_t)N, (cuuint64_t)T * R}, s[1] = {(cuuint64_t)N * 2};
    const cuuint32_t box[2] = {64, ROWS};
    M.q = make_map(Q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, s, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.k = make_map(K.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, s, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint32_t box[3] = {P, 1, C};
    const cuuint64_t d[3] = {(cuuint64_t)P, (cuuint64_t)H, (cuuint64_t)T};
    const cuuint64_t sv[2] = {(cuuint64_t)P * 2, (cuuint64_t)V.stride(0) * 2};
    const cuuint64_t sd[2] = {(cuuint64_t)P * 2, (cuuint64_t)DY.stride(0) * 2};
    const cuuint64_t sz[2] = {(cuuint64_t)P * 2, (cuuint64_t)Z.stride(0) * 2};
    M.v = make_map(V.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, sv, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.dy = make_map(DY.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, sd, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.z = make_map(Z.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, sz, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint64_t d[3] = {(cuuint64_t)NQ, (cuuint64_t)H, (cuuint64_t)T}, s[2] = {(cuuint64_t)NQ * 4, (cuuint64_t)H * NQ * 4};
    const cuuint32_t box[3] = {NQ, 1, C};
    M.ang = make_map(ang.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, d, s, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  M.sth = hs ? state_map(*hs) : M.q;
  M.std_ = ds ? state_map(*ds) : M.q;
  return M;
}

static void launch2(const Maps& M, const Args& a, int dev) {
  const int smem = (int)sizeof(Smem) + 1024;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(bwd_tc2_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "bwd_tc2: shared memory request too large: ", smem);
    attr = true;
  }
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  if (a.NB == 0) return;
  const int grid = std::min(a.NB, nsm);
  bwd_tc2_kernel<1><<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(M, a);
}

static unsigned long long* g_prof = nullptr;
void set_prof(torch::Tensor t) { g_prof = t.numel() ? reinterpret_cast<unsigned long long*>(t.data_ptr()) : nullptr; }

static Args base_args(torch::Tensor V, torch::Tensor qb, torch::Tensor kb, torch::Tensor mv, torch::Tensor mo, torch::Tensor mz,
                      torch::Tensor Dskip, torch::Tensor ell, torch::Tensor ff, torch::Tensor gam, torch::Tensor tot,
                      torch::Tensor sb_start, torch::Tensor sb_len, torch::Tensor ain, torch::Tensor sb_chunk, int64_t NB) {
  TORCH_CHECK(ain.is_contiguous() && ain.scalar_type() == torch::kFloat && sb_chunk.scalar_type() == torch::kInt);
  TORCH_CHECK(qb.scalar_type() == torch::kBFloat16 && kb.scalar_type() == torch::kBFloat16 && qb.is_contiguous() && kb.is_contiguous());
  Args a{};
  a.qbias = reinterpret_cast<const __nv_bfloat16*>(qb.data_ptr()); a.kbias = reinterpret_cast<const __nv_bfloat16*>(kb.data_ptr());
  a.mv = mv.data_ptr<float>(); a.mo = mo.data_ptr<float>();
  a.mz = mz.data_ptr<float>(); a.Dskip = Dskip.data_ptr<float>(); a.ell = ell.data_ptr<float>(); a.ff = ff.data_ptr<float>();
  a.gam = gam.data_ptr<float>(); a.tot = tot.data_ptr<float>(); a.sb_start = sb_start.data_ptr<int>();
  a.sb_len = sb_len.data_ptr<int>(); a.S = V.size(0); a.H = V.size(1); a.NB = (int)NB; a.prof = g_prof;
  a.ain = ain.data_ptr<float>(); a.sb_chunk = sb_chunk.data_ptr<int>();
  return a;
}

void bwd_main(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor DY, torch::Tensor Z, torch::Tensor ang,
              torch::Tensor qb, torch::Tensor kb, torch::Tensor mv, torch::Tensor mo, torch::Tensor mz, torch::Tensor Dskip,
              torch::Tensor ell, torch::Tensor ff, torch::Tensor gam, torch::Tensor tot, torch::Tensor sb_start,
              torch::Tensor sb_len, torch::Tensor ain, torch::Tensor sb_chunk, torch::Tensor hs, torch::Tensor ds, torch::Tensor dq, torch::Tensor dk, torch::Tensor dz,
              torch::Tensor dv, torch::Tensor dth, torch::Tensor dlrow, torch::Tensor dgam,
              torch::Tensor dfac, torch::Tensor accq, torch::Tensor acck, torch::Tensor cross, int64_t NB) {
  Args a = base_args(V, qb, kb, mv, mo, mz, Dskip, ell, ff, gam, tot, sb_start, sb_len, ain, sb_chunk, NB);
  a.dq = reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()); a.dk = reinterpret_cast<__nv_bfloat16*>(dk.data_ptr());
  a.dz = reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()); a.dv = reinterpret_cast<__nv_bfloat16*>(dv.data_ptr());
  a.dth = dth.data_ptr<float>();
  a.dlrow = dlrow.data_ptr<float>(); a.dgam = dgam.data_ptr<float>(); a.dfac = dfac.data_ptr<float>();
  a.accq = accq.data_ptr<float>(); a.acck = acck.data_ptr<float>(); a.cross = cross.data_ptr<float>();
  launch2(build_maps(Q, K, V, DY, Z, ang, &hs, &ds), a, V.get_device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("main", &bwd_main);
  m.def("set_prof", &set_prof);
  m.def("nacc", []() { return NACC; });
  m.def("smem", []() { return (int64_t)sizeof(Smem) + 1024; });
  m.def("grid", [](int64_t NB) { int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0); return std::min<int64_t>(NB, nsm); });
}
