/**
 *  Copyright (c) 2023 by MPI-SWS DSG
 * @file graph/serialize/mmap_file.h
 * @brief mmap a file as a data pointer
 */
#ifndef DGL_GRAPH_SERIALIZE_MMAP_FILE_H_
#define DGL_GRAPH_SERIALIZE_MMAP_FILE_H_

#include <fcntl.h>
#include <sys/mman.h>  // mmap() is defined in this header
#include <sys/stat.h>  //for fstat
#include <unistd.h>    // ftruncate

namespace dgl {
namespace serialize {

struct MmapFile {
  explicit MmapFile(std::string path_) {
    path = path_;
    fd = open(path.c_str(), O_RDWR, 0x7777);
    CHECK_GE(fd, 0) << "Can't open file: " << path;
    struct stat st;
    CHECK_EQ(fstat(fd, &st), 0) << "";
    len = st.st_size;
    data_ptr = mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK_NE(data_ptr, MAP_FAILED) << "mmap failed for file: " << path;
  }

  ~MmapFile() {
    CHECK_EQ(munmap(data_ptr, len), 0) << "munmap failed for file: " << path;
    CHECK_EQ(close(fd), 0) << "Can't close file: " << path;
  }

  inline uint32_t* AsUint32Ptr() {
    return reinterpret_cast<uint32_t*>(data_ptr);
  }
  inline size_t GetLength() { return len; }

 private:
  int fd;
  size_t len;
  void* data_ptr;
  std::string path;
};

}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_SERIALIZE_MMAP_FILE_H_
