// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#if defined(__linux__) && IS_NVIDIA_GPU && defined(USE_NCCL)

namespace host_shm {

struct File {
  int fd;
  explicit File(int fd) : fd(fd) {}
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File() {
    if (fd >= 0 && close(fd) != 0)
      std::fprintf(stderr, "Tutel host SHM close: %s\n", std::strerror(errno));
  }
};

struct Region {
  void* host = nullptr;
  void* device = nullptr;
  bool registered = false;
};

struct Descriptor {
  char path[64];
  uint64_t file_device;
  uint64_t file_inode;
  int64_t bytes;
  int64_t numel;
  int64_t dtype;
};

struct Allocation {
  int device_index;
  size_t bytes;
  std::vector<Region> regions;
  std::string path = "/dev/shm/tutel-host-XXXXXX";
  bool owns_file = false;
  void* pointers = nullptr;

  Allocation(int device_index, size_t bytes, int ranks)
      : device_index(device_index), bytes(bytes), regions(ranks) {}

  static bool cuda_cleanup(cudaError_t error, const char* operation) noexcept {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "Tutel host SHM %s: %s\n", operation, cudaGetErrorString(error));
    return false;
  }

  ~Allocation() {
    if (owns_file && unlink(path.c_str()) != 0)
      std::fprintf(stderr, "Tutel host SHM unlink: %s\n", std::strerror(errno));

    bool gpu_resources = pointers != nullptr;
    for (const auto& region : regions)
      gpu_resources = gpu_resources || region.registered;

    bool safe_to_unmap = true;
    int previous_device = -1;
    if (gpu_resources) {
      safe_to_unmap = cuda_cleanup(cudaGetDevice(&previous_device), "cudaGetDevice");
      if (safe_to_unmap)
        safe_to_unmap = cuda_cleanup(cudaSetDevice(device_index), "cudaSetDevice");
      // Raw mapped-host pointers may have been used on any local CUDA stream.
      if (safe_to_unmap)
        safe_to_unmap = cuda_cleanup(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
      if (safe_to_unmap) {
        if (pointers) cuda_cleanup(cudaFree(pointers), "cudaFree pointer table");
        for (auto& region : regions) {
          if (region.registered && cuda_cleanup(cudaHostUnregister(region.host), "cudaHostUnregister"))
            region.registered = false;
        }
      } else {
        std::fprintf(stderr, "Tutel host SHM: retaining mappings because CUDA cleanup could not complete safely\n");
      }
      if (previous_device >= 0 && previous_device != device_index)
        cuda_cleanup(cudaSetDevice(previous_device), "restore CUDA device");
    }
    for (const auto& region : regions) {
      if (safe_to_unmap && region.host && !region.registered && munmap(region.host, bytes) != 0)
        std::fprintf(stderr, "Tutel host SHM munmap: %s\n", std::strerror(errno));
    }
  }

  void map(int rank, int fd) {
    void* address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    TORCH_CHECK(address != MAP_FAILED, "Tutel host SHM mmap: ", std::strerror(errno));
    regions[rank].host = address;
  }
};

} // namespace host_shm

// Collective over the Gloo group installed by put_pg_storage(0, group).
// Returns (CUDA view of local pinned host memory, CUDA int64 peer pointers).
// The pointer table is indexed by group-local rank; neither view moves host data.
// All ranks must share /dev/shm and use the same shape/dtype. Allocate outside
// graph capture. Keep at least one original returned tensor alive while using
// the mappings; pointer-table clones and CUDA graphs do not retain their owner.
// Like uncached_empty_ex, the allocation includes a zeroed 64 KiB trailing area.
// "Uncached" names the non-caching allocator, not a CPU cache-disable guarantee.
static std::tuple<torch::Tensor, torch::Tensor> uncached_empty_ex_host(torch::IntArrayRef shape, at::ScalarType dtype) {
#if defined(__linux__) && IS_NVIDIA_GPU
  auto it = _pg_storage.find(0);
  TORCH_CHECK(it != _pg_storage.end() && it->second.get() != nullptr,
      "uncached_empty_ex_host requires put_pg_storage(0, gloo_group)");
  auto pg = it->second;
  TORCH_CHECK(pg->getBackend(c10::DeviceType::CPU)->getBackendName() == "gloo",
      "uncached_empty_ex_host requires a Gloo CPU backend");
  const int rank = pg->getRank(), ranks = pg->getSize();
  TORCH_CHECK(rank >= 0 && rank < ranks, "Invalid process-group rank");
  TORCH_CHECK(!c10::isQIntType(dtype), "uncached_empty_ex_host does not support quantized dtypes");

  int64_t numel = 1;
  const int64_t limit = std::numeric_limits<int64_t>::max();
  for (int64_t dim : shape) {
    TORCH_CHECK(dim >= 0, "uncached_empty_ex_host requires nonnegative dimensions");
    TORCH_CHECK(numel == 0 || dim <= limit / numel, "Host tensor size overflow");
    numel *= dim;
  }
  const int64_t item_size = torch::elementSize(dtype);
  const int64_t padding = 64 * 1024;
  const long page_size = sysconf(_SC_PAGESIZE);
  TORCH_CHECK(page_size > 0, "Cannot determine host page size");
  TORCH_CHECK(item_size > 0 && numel <= (limit - padding - page_size) / item_size,
      "Host tensor allocation size overflow");
  const int64_t bytes = ((numel * item_size + padding + page_size - 1) / page_size) * page_size;
  TORCH_CHECK(static_cast<uint64_t>(bytes) <= std::numeric_limits<size_t>::max() &&
      static_cast<uint64_t>(bytes) <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()),
      "Host tensor allocation exceeds the platform size limit");

  const int device_index = c10::cuda::current_device();
  const at::Device device(at::DeviceType::CUDA, device_index);
  at::DeviceGuard device_guard(device);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus capture;
  AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
      "uncached_empty_ex_host must be called outside CUDA graph capture");

  auto owner = std::make_shared<host_shm::Allocation>(device_index, static_cast<size_t>(bytes), ranks);
  host_shm::File local_file(mkstemp(&owner->path[0]));
  owner->owns_file = local_file.fd >= 0;
  TORCH_CHECK(owner->owns_file, "Cannot create host SHM file: ", std::strerror(errno));
  TORCH_CHECK(fcntl(local_file.fd, F_SETFD, FD_CLOEXEC) != -1,
      "Cannot set host SHM close-on-exec: ", std::strerror(errno));
  // Reserve tmpfs space before touching pages, avoiding a later SIGBUS on ENOSPC.
  int allocation_error = posix_fallocate(local_file.fd, 0, static_cast<off_t>(bytes));
  TORCH_CHECK(allocation_error == 0, "Cannot allocate host SHM: ", std::strerror(allocation_error));
  owner->map(rank, local_file.fd);
  std::memset(owner->regions[rank].host, 0, owner->bytes);

  struct stat local_stat;
  TORCH_CHECK(fstat(local_file.fd, &local_stat) == 0,
      "Cannot stat local host SHM: ", std::strerror(errno));
  host_shm::Descriptor descriptor{};
  TORCH_CHECK(owner->path.size() < sizeof(descriptor.path), "Host SHM path is too long");
  std::memcpy(descriptor.path, owner->path.c_str(), owner->path.size() + 1);
  descriptor.file_device = static_cast<uint64_t>(local_stat.st_dev);
  descriptor.file_inode = static_cast<uint64_t>(local_stat.st_ino);
  descriptor.bytes = bytes;
  descriptor.numel = numel;
  descriptor.dtype = static_cast<int64_t>(dtype);
  auto metadata = torch::empty({1, static_cast<int64_t>(sizeof(descriptor))},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  std::memcpy(metadata.data_ptr(), &descriptor, sizeof(descriptor));
  auto descriptors = all_gather_native(metadata);
  TORCH_CHECK(descriptors.size(0) == ranks, "Unexpected host SHM descriptor count");

  auto cpu_pointers = torch::empty({ranks}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto pointer_data = cpu_pointers.data_ptr<int64_t>();
  for (int peer = 0; peer < ranks; ++peer) {
    host_shm::Descriptor remote{};
    std::memcpy(&remote, descriptors[peer].data_ptr(), sizeof(remote));
    TORCH_CHECK(remote.bytes == bytes && remote.numel == numel && remote.dtype == descriptor.dtype,
        "All ranks must use matching host tensor sizes and dtypes");
    TORCH_CHECK(std::memchr(remote.path, '\0', sizeof(remote.path)) != nullptr &&
        std::strncmp(remote.path, "/dev/shm/tutel-host-", sizeof("/dev/shm/tutel-host-") - 1) == 0,
        "Invalid host SHM descriptor path");
    if (peer != rank) {
      host_shm::File peer_file(open(remote.path, O_RDWR | O_CLOEXEC | O_NOFOLLOW));
      TORCH_CHECK(peer_file.fd >= 0, "Cannot open rank ", peer,
          " host SHM; all ranks must share the same host and /dev/shm namespace: ", std::strerror(errno));
      struct stat peer_stat;
      TORCH_CHECK(fstat(peer_file.fd, &peer_stat) == 0,
          "Cannot stat peer host SHM: ", std::strerror(errno));
      TORCH_CHECK(S_ISREG(peer_stat.st_mode) && peer_stat.st_size == bytes &&
          static_cast<uint64_t>(peer_stat.st_dev) == remote.file_device &&
          static_cast<uint64_t>(peer_stat.st_ino) == remote.file_inode,
          "Peer host SHM identity/size mismatch; a shared /dev/shm namespace is required");
      owner->map(peer, peer_file.fd);
    }
    auto& region = owner->regions[peer];
    AT_CUDA_CHECK(cudaHostRegister(region.host, owner->bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
    region.registered = true;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&region.device, region.host, 0));
    // The device alias need not equal the CPU mapping's virtual address.
    pointer_data[peer] = reinterpret_cast<int64_t>(region.device);
  }

  auto connected = pg->barrier();
  TORCH_CHECK(connected->wait(), "Host SHM connection barrier failed");
  TORCH_CHECK(unlink(owner->path.c_str()) == 0,
      "Cannot unlink connected host SHM file: ", std::strerror(errno));
  owner->owns_file = false;

  const size_t pointer_bytes = static_cast<size_t>(ranks) * sizeof(int64_t);
  AT_CUDA_CHECK(cudaMalloc(&owner->pointers, pointer_bytes));
  AT_CUDA_CHECK(cudaMemcpyAsync(owner->pointers, pointer_data, pointer_bytes, cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  // Explicit target_device avoids inferring a device from the host allocation.
  auto local = at::from_blob(owner->regions[rank].device, shape,
      [owner](void*) {}, torch::TensorOptions().dtype(dtype).device(device), device);
  auto pointers = torch::from_blob(owner->pointers, {ranks},
      [owner](void*) {}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  return {local, pointers};
#else
  TORCH_CHECK(false, "uncached_empty_ex_host currently requires Linux and NVIDIA CUDA");
#endif
}

TORCH_LIBRARY_FRAGMENT(tutel_ops, m) {
  m.def("uncached_empty_ex_host", uncached_empty_ex_host);
}
#endif
