#include <torch/torch.h>



// CUDA declarations
at::Tensor float2gemmlowp(at::Tensor in, float range, float offset, int num_bits, bool int_exp,
                          bool enforce_true_zero, at::Tensor noise);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float2gemmlowp", &float2gemmlowp, "Convert float 32 to gemmlowp");
}
