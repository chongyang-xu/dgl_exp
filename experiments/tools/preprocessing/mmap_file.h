/**
 *  Copyright (c) 2023 by MPI-SWS DSG
 * @brief mmap a file as a data pointer
 */
#ifndef _MMAP_FILE_H_
#define _MMAP_FILE_H_

#include <fcntl.h>
#include <sys/mman.h>  // mmap() is defined in this header
#include <sys/stat.h>  //for fstat
#include <unistd.h>    // ftruncate

#include <cassert>
#include <string>
#include <iostream>

#define CHECK_GE(a, b) if ( (a) <  (b)) std::cout 
#define CHECK_NE(a, b) if ( (a) == (b)) std::cout 
#define CHECK_EQ(a, b) if ( (a) != (b)) std::cout 

struct MmapFile {
  explicit MmapFile(std::string path_) {
    path = path_;
    fd = open(path.c_str(), O_RDONLY, 0x4440);
    CHECK_GE(fd, 0) << "Can't open file: " << path << std::endl;
    struct stat st;
    fstat(fd, &st);
    len = st.st_size;
    data_ptr = mmap(0, len, PROT_READ, MAP_SHARED, fd, 0);
    CHECK_NE(data_ptr, MAP_FAILED) << "mmap failed for file: " << path << std::endl;
  }

  explicit MmapFile(std::string path_, size_t create_length) {
    path = path_;
    fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0x7777);
    len = create_length;
    ftruncate(fd, len);
    data_ptr = mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK_NE(data_ptr, MAP_FAILED) << "mmap failed for file: " << path << std::endl;
  }

  ~MmapFile() {
    CHECK_EQ(munmap(data_ptr, len), 0) << "munmap failed for file: " << path << std::endl;
    CHECK_EQ(close(fd), 0) << "Can't close file: " << path << std::endl;
  }

  inline int64_t* AsInt64Ptr() {
    return reinterpret_cast<int64_t*>(data_ptr);
  }
  inline float* AsFloat32Ptr() {
    return reinterpret_cast<float*>(data_ptr);
  }
  inline uint32_t* AsUint32Ptr() {
    return reinterpret_cast<uint32_t*>(data_ptr);
  }
  inline uint8_t* AsUint8Ptr() {
    return reinterpret_cast<uint8_t*>(data_ptr);
  }
  inline int8_t* AsInt8Ptr() {
    return reinterpret_cast<int8_t*>(data_ptr);
  }
  inline size_t GetLength() { return len; }

 private:
  int fd;
  size_t len;
  void* data_ptr;
  std::string path;
};


#endif  // _MMAP_FILE_H_
