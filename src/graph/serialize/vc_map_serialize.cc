/**
 *  Copyright (c) 2023 by MPI-SWS DSG
 * @file graph/serialize/unordered_map_serialize.cc
 * @brief Graph serialization implementation
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>

#include "../../c_api_common.h"

using namespace dgl::runtime;
using dmlc::SeekStream;

namespace dgl {
namespace serialize {

constexpr uint64_t kDGLSerialize_VCMap = 0xDD5A9FBE3FA24440;

DGL_REGISTER_GLOBAL("data.tensor_serialize._CAPI_SaveVCMap")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      auto fs = std::unique_ptr<dmlc::Stream>(
          dmlc::Stream::Create(filename.c_str(), "w"));
      CHECK(fs) << "Filename is invalid";
      fs->Write(kDGLSerialize_VCMap);
      LOG(FATAL) << "not implemented";
      *rv = true;
    });

DGL_REGISTER_GLOBAL("data.tensor_serialize._CAPI_LoadVCMap")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      auto fs = std::unique_ptr<dmlc::Stream>(
          dmlc::Stream::Create(filename.c_str(), "r"));
      CHECK(fs) << "Filename is invalid or file doesn't exists";
      uint64_t magincNum;
      CHECK(fs->Read(&magincNum)) << "Invalid file";
      CHECK_EQ(magincNum, kDGLSerialize_VCMap) << "Invalid DGL tensor file";
      
      LOG(FATAL) << "not implemented";
      *rv = r;
    });

}  // namespace serialize
}  // namespace dgl
