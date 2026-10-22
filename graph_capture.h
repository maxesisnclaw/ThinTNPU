#pragma once
#include <ATen/ATen.h>
#include <acl/acl_rt.h>
#include <memory>
#include <vector>

namespace thin {
struct CapturedAllocation {
  void* pointer;
  explicit CapturedAllocation(void* p) : pointer(p) {}
  ~CapturedAllocation();
};
struct CapturedHostAllocation {
  void* pointer;
  explicit CapturedHostAllocation(void* p) : pointer(p) {}
  ~CapturedHostAllocation();
};
struct CaptureMemory {
  std::vector<std::shared_ptr<CapturedAllocation>> allocations;
  std::vector<std::shared_ptr<CapturedHostAllocation>> host_inputs;
};
extern thread_local CaptureMemory* capture_memory;
inline bool capturing() { return capture_memory != nullptr; }
}
