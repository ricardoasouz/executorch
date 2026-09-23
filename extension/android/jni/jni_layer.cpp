/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

#include <executorch/extension/android/jni/jni_helper.h>
#include <executorch/extension/android/jni/jni_layer_constants.h>

#include <executorch/extension/android/jni/log.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <chrono>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef ET_USE_THREADPOOL
#include <cpuinfo.h>
#include <executorch/extension/threadpool/threadpool.h>
#ifdef EXECUTORCH_HAS_THREADPOOL_USE_N_THREADS_GUARD
#include <executorch/extension/threadpool/fb/threadpool_use_n_threads.h>
#endif
#endif

#ifdef EXECUTORCH_ANDROID_PROFILING
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

using namespace executorch::extension;
using namespace torch::executor;

namespace executorch::extension {
class JTensor : public facebook::jni::JavaClass<JTensor> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/Tensor;";

  static facebook::jni::local_ref<JTensor::javaobject> newJTensorFromTensor(
      const executorch::aten::Tensor& tensor) {
    // Java wrapper currently only supports contiguous tensors.

    const auto scalarType = tensor.scalar_type();
    if (scalar_type_to_java_dtype.count(scalarType) == 0) {
      std::stringstream ss;
      ss << "executorch::aten::Tensor scalar type "
         << static_cast<int>(scalarType) << " is not supported on java side";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
      return nullptr;
    }
    int jdtype = scalar_type_to_java_dtype.at(scalarType);

    const auto& tensor_shape = tensor.sizes();
    std::vector<jlong> tensor_shape_vec;
    for (const auto& s : tensor_shape) {
      tensor_shape_vec.push_back(s);
    }
    facebook::jni::local_ref<jlongArray> jTensorShape =
        facebook::jni::make_long_array(tensor_shape_vec.size());
    jTensorShape->setRegion(
        0, tensor_shape_vec.size(), tensor_shape_vec.data());

    static auto cls = JTensor::javaClassStatic();
    // Note: this is safe as long as the data stored in tensor is valid; the
    // data won't go out of scope as long as the Method for the inference is
    // valid and there is no other inference call. Java layer picks up this
    // value immediately so the data is valid.
    facebook::jni::local_ref<facebook::jni::JByteBuffer> jTensorBuffer =
        facebook::jni::JByteBuffer::wrapBytes(
            (uint8_t*)tensor.data_ptr(), tensor.nbytes());
    jTensorBuffer->order(facebook::jni::JByteOrder::nativeOrder());

    static const auto jMethodNewTensor =
        cls->getStaticMethod<facebook::jni::local_ref<JTensor::javaobject>(
            facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
            facebook::jni::alias_ref<jlongArray>,
            jint)>("nativeNewTensor");
    return jMethodNewTensor(cls, jTensorBuffer, jTensorShape, jdtype);
  }

  static TensorPtr newTensorFromJTensor(
      facebook::jni::alias_ref<JTensor::javaobject> jtensor) {
    static auto cls = JTensor::javaClassStatic();
    static const auto dtypeMethod = cls->getMethod<jint()>("dtypeJniCode");
    jint jdtype = dtypeMethod(jtensor);

    static const auto shapeField = cls->getField<jlongArray>("shape");
    auto jshape = jtensor->getFieldValue(shapeField);

    static auto dataBufferMethod = cls->getMethod<
        facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
        "getRawDataBuffer");
    facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
        dataBufferMethod(jtensor);

    const auto rank = jshape->size();

    const auto shapeArr = jshape->getRegion(0, rank);
    std::vector<executorch::aten::SizesType> shape_vec;
    shape_vec.reserve(rank);

    int64_t numel = 1;
    for (int i = 0; i < rank; i++) {
      shape_vec.push_back(shapeArr[i]);
    }
    for (int i = rank - 1; i >= 0; --i) {
      numel *= shapeArr[i];
    }
    JNIEnv* jni = facebook::jni::Environment::current();
    if (java_dtype_to_scalar_type.count(jdtype) == 0) {
      std::stringstream ss;
      ss << "Unknown Tensor jdtype: [" << jdtype << "]";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
      return nullptr;
    }
    ScalarType scalar_type = java_dtype_to_scalar_type.at(jdtype);
    const jlong dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
    if (dataCapacity < 0) {
      std::stringstream ss;
      ss << "Tensor buffer is not direct or has invalid capacity";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
      return nullptr;
    }
    const size_t elementSize = executorch::runtime::elementSize(scalar_type);
    const jlong expectedElements = static_cast<jlong>(numel);
    const jlong expectedBytes =
        expectedElements * static_cast<jlong>(elementSize);
    const bool matchesElements = dataCapacity == expectedElements;
    const bool matchesBytes = dataCapacity == expectedBytes;
    if (!matchesElements && !matchesBytes) {
      std::stringstream ss;
      ss << "Tensor dimensions(elements number: " << numel
         << ") inconsistent with buffer capacity " << dataCapacity
         << " (element size bytes: " << elementSize << ")";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
      return nullptr;
    }
    return from_blob(
        jni->GetDirectBufferAddress(jbuffer.get()), shape_vec, scalar_type);
  }
};

class JEValue : public facebook::jni::JavaClass<JEValue> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/EValue;";

  constexpr static int kTypeCodeTensor = 1;
  constexpr static int kTypeCodeString = 2;
  constexpr static int kTypeCodeDouble = 3;
  constexpr static int kTypeCodeInt = 4;
  constexpr static int kTypeCodeBool = 5;

  static facebook::jni::local_ref<JEValue> newJEValueFromEValue(EValue evalue) {
    if (evalue.isTensor()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                  facebook::jni::local_ref<JTensor::javaobject>)>("from");
      return jMethodTensor(
          JEValue::javaClassStatic(),
          JTensor::newJTensorFromTensor(evalue.toTensor()));
    } else if (evalue.isInt()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jlong)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toInt());
    } else if (evalue.isDouble()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jdouble)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toDouble());
    } else if (evalue.isBool()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jboolean)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toBool());
    } else if (evalue.isString()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                  facebook::jni::local_ref<jstring>)>("from");
      std::string str =
          std::string(evalue.toString().begin(), evalue.toString().end());
      return jMethodTensor(
          JEValue::javaClassStatic(), facebook::jni::make_jstring(str));
    }
    std::stringstream ss;
    ss << "Unknown EValue type: [" << static_cast<int>(evalue.tag) << "]";
    jni_helper::throwExecutorchException(
        static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    return {};
  }

  static TensorPtr JEValueToTensorImpl(
      facebook::jni::alias_ref<JEValue> JEValue) {
    static const auto typeCodeField =
        JEValue::javaClassStatic()->getField<jint>("mTypeCode");
    const auto typeCode = JEValue->getFieldValue(typeCodeField);
    if (JEValue::kTypeCodeTensor == typeCode) {
      static const auto jMethodGetTensor =
          JEValue::javaClassStatic()
              ->getMethod<facebook::jni::alias_ref<JTensor::javaobject>()>(
                  "toTensor");
      auto jtensor = jMethodGetTensor(JEValue);
      return JTensor::newTensorFromJTensor(jtensor);
    }
    std::stringstream ss;
    ss << "Unknown EValue typeCode: " << typeCode;
    jni_helper::throwExecutorchException(
        static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    return {};
  }
};

#ifdef EXECUTORCH_BUILD_EXTENSION_TRAINING
// Training still uses fbjni and shares these conversions with this translation
// unit. Keep the bridge out of line until TrainingModule and SGD migrate to
// standard JNI as a later step of #10444.
namespace training_jni_compat {

facebook::jni::local_ref<JTensor::javaobject> new_jtensor_from_tensor(
    const executorch::aten::Tensor& tensor) {
  return JTensor::newJTensorFromTensor(tensor);
}

TensorPtr new_tensor_from_jtensor(
    facebook::jni::alias_ref<JTensor::javaobject> tensor) {
  return JTensor::newTensorFromJTensor(tensor);
}

facebook::jni::local_ref<JEValue> new_jevalue_from_evalue(EValue value) {
  return JEValue::newJEValueFromEValue(value);
}

TensorPtr new_tensor_from_jevalue(facebook::jni::alias_ref<JEValue> value) {
  return JEValue::JEValueToTensorImpl(value);
}

} // namespace training_jni_compat
#endif

class ExecuTorchJni {
 private:
  std::unique_ptr<Module> module_;
#if defined(ET_USE_THREADPOOL) && \
    defined(EXECUTORCH_HAS_THREADPOOL_USE_N_THREADS_GUARD)
  int num_threads_{0};
#endif

 public:
  ExecuTorchJni(
      const std::string& model_path,
      jint load_mode_value,
      jint num_threads) {
    Module::LoadMode load_mode = Module::LoadMode::Mmap;
    if (load_mode_value == 0) {
      load_mode = Module::LoadMode::File;
    } else if (load_mode_value == 2) {
      load_mode = Module::LoadMode::MmapUseMlock;
    } else if (load_mode_value == 3) {
      load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors;
    }
#ifdef EXECUTORCH_ANDROID_PROFILING
    auto etdump_gen = std::make_unique<executorch::etdump::ETDumpGen>();
#else
    auto etdump_gen = nullptr;
#endif
    module_ =
        std::make_unique<Module>(model_path, load_mode, std::move(etdump_gen));

#ifdef ET_USE_THREADPOOL
    int thread_count =
        num_threads != 0 ? num_threads : cpuinfo_get_processors_count() / 2;
#ifdef EXECUTORCH_HAS_THREADPOOL_USE_N_THREADS_GUARD
    num_threads_ = thread_count;
#else
    auto threadpool = executorch::extension::threadpool::get_threadpool();
    if (threadpool && thread_count > 0) {
      threadpool->_unsafe_reset_threadpool(thread_count);
    }
#endif
#endif
  }

  Error load_with_options(
      const std::vector<std::string>& backend_names,
      const std::vector<std::string>& option_keys,
      const std::vector<jint>& option_values,
      std::string& error_message) {
    if (option_keys.size() != backend_names.size() ||
        option_values.size() != backend_names.size()) {
      error_message = "backend option arrays must have equal length";
      return Error::InvalidArgument;
    }

    std::map<std::string, std::vector<executorch::runtime::BackendOption>>
        grouped;
    for (size_t i = 0; i < backend_names.size(); ++i) {
      executorch::runtime::BackendOption option{};
      const auto& key = option_keys[i];
      if (key.size() >= executorch::runtime::kMaxOptionKeyLength) {
        error_message = "backend option key too long: " + key;
        return Error::InvalidArgument;
      }
      std::strncpy(
          option.key,
          key.c_str(),
          executorch::runtime::kMaxOptionKeyLength - 1);
      option.key[executorch::runtime::kMaxOptionKeyLength - 1] = '\0';
      option.value = static_cast<int>(option_values[i]);
      grouped[backend_names[i]].push_back(option);
    }

    executorch::runtime::LoadBackendOptionsMap options_map;
    for (auto& entry : grouped) {
      const auto error = options_map.set_options(
          entry.first.c_str(),
          executorch::runtime::Span<executorch::runtime::BackendOption>(
              entry.second.data(), entry.second.size()));
      if (error != Error::Ok) {
        error_message = "Failed to set backend options for " + entry.first;
        return error;
      }
    }

    const auto error = module_->load(options_map);
    if (error != Error::Ok) {
      error_message = "Failed to load Module with backend options";
    }
    return error;
  }

  Module* module() const {
    return module_.get();
  }

  Method* get_method(const std::string& method_name) {
    const auto it = module_->methods_.find(method_name);
    return it == module_->methods_.end() ? nullptr : it->second.method.get();
  }

#if defined(ET_USE_THREADPOOL) && \
    defined(EXECUTORCH_HAS_THREADPOOL_USE_N_THREADS_GUARD)
  int num_threads() const {
    return num_threads_;
  }
#endif

  jboolean etdump() {
    return etdump_to_path("/data/local/tmp/result.etdump");
  }

  jboolean etdump_to(const std::string& output_path) {
    return etdump_to_path(output_path.c_str());
  }

 private:
  jboolean etdump_to_path(const char* path) {
#ifdef EXECUTORCH_ANDROID_PROFILING
    auto* tracer = module_->event_tracer();
    if (!tracer) {
      ET_LOG(Error, "ETDump not available: no event tracer attached");
      return false;
    }
    auto* etdumpgen = static_cast<executorch::etdump::ETDumpGen*>(tracer);
    auto etdump_data = etdumpgen->get_etdump_data();

    if (etdump_data.buf != nullptr && etdump_data.size > 0) {
      int etdump_file = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (etdump_file == -1) {
        ET_LOG(Error, "Cannot create %s error: %d", path, errno);
        free(etdump_data.buf);
        return false;
      }
      ssize_t bytes_written =
          write(etdump_file, (uint8_t*)etdump_data.buf, etdump_data.size);
      if (bytes_written == -1) {
        ET_LOG(Error, "Cannot write %s error: %d", path, errno);
        close(etdump_file);
        free(etdump_data.buf);
        return false;
      } else {
        ET_LOG(Info, "ETDump written %zd bytes to %s.", bytes_written, path);
      }
      close(etdump_file);
      free(etdump_data.buf);
      return true;
    } else {
      ET_LOG(Error, "No ETDump data available!");
    }
#else
    (void)path;
#endif
    return false;
  }
};
} // namespace executorch::extension

namespace {

constexpr int kTypeCodeTensor = 1;
constexpr int kTypeCodeDouble = 3;
constexpr int kTypeCodeInt = 4;
constexpr int kTypeCodeBool = 5;

struct RawJniCache {
  jclass tensor_class{nullptr};
  jclass evalue_class{nullptr};
  jclass string_class{nullptr};
  jclass byte_buffer_class{nullptr};
  jclass byte_order_class{nullptr};

  jmethodID tensor_new{nullptr};
  jmethodID tensor_dtype{nullptr};
  jmethodID tensor_data_buffer{nullptr};
  jfieldID tensor_shape{nullptr};

  jmethodID evalue_from_tensor{nullptr};
  jmethodID evalue_from_long{nullptr};
  jmethodID evalue_from_double{nullptr};
  jmethodID evalue_from_bool{nullptr};
  jmethodID evalue_from_string{nullptr};
  jmethodID evalue_to_tensor{nullptr};
  jmethodID evalue_to_long{nullptr};
  jmethodID evalue_to_double{nullptr};
  jmethodID evalue_to_bool{nullptr};
  jfieldID evalue_type_code{nullptr};

  jmethodID byte_buffer_order{nullptr};
  jmethodID byte_order_native{nullptr};

  bool initialize(JNIEnv* env) {
    tensor_class = find_global_class(env, "org/pytorch/executorch/Tensor");
    evalue_class = find_global_class(env, "org/pytorch/executorch/EValue");
    string_class = find_global_class(env, "java/lang/String");
    byte_buffer_class = find_global_class(env, "java/nio/ByteBuffer");
    byte_order_class = find_global_class(env, "java/nio/ByteOrder");
    if (env->ExceptionCheck() || tensor_class == nullptr ||
        evalue_class == nullptr || string_class == nullptr ||
        byte_buffer_class == nullptr || byte_order_class == nullptr) {
      return false;
    }

    tensor_new = env->GetStaticMethodID(
        tensor_class,
        "nativeNewTensor",
        "(Ljava/nio/ByteBuffer;[JI)Lorg/pytorch/executorch/Tensor;");
    tensor_dtype = env->GetMethodID(tensor_class, "dtypeJniCode", "()I");
    tensor_data_buffer = env->GetMethodID(
        tensor_class, "getRawDataBuffer", "()Ljava/nio/Buffer;");
    tensor_shape = env->GetFieldID(tensor_class, "shape", "[J");

    evalue_from_tensor = env->GetStaticMethodID(
        evalue_class,
        "from",
        "(Lorg/pytorch/executorch/Tensor;)Lorg/pytorch/executorch/EValue;");
    evalue_from_long = env->GetStaticMethodID(
        evalue_class, "from", "(J)Lorg/pytorch/executorch/EValue;");
    evalue_from_double = env->GetStaticMethodID(
        evalue_class, "from", "(D)Lorg/pytorch/executorch/EValue;");
    evalue_from_bool = env->GetStaticMethodID(
        evalue_class, "from", "(Z)Lorg/pytorch/executorch/EValue;");
    evalue_from_string = env->GetStaticMethodID(
        evalue_class,
        "from",
        "(Ljava/lang/String;)Lorg/pytorch/executorch/EValue;");
    evalue_to_tensor = env->GetMethodID(
        evalue_class, "toTensor", "()Lorg/pytorch/executorch/Tensor;");
    evalue_to_long = env->GetMethodID(evalue_class, "toInt", "()J");
    evalue_to_double = env->GetMethodID(evalue_class, "toDouble", "()D");
    evalue_to_bool = env->GetMethodID(evalue_class, "toBool", "()Z");
    evalue_type_code = env->GetFieldID(evalue_class, "mTypeCode", "I");

    byte_buffer_order = env->GetMethodID(
        byte_buffer_class,
        "order",
        "(Ljava/nio/ByteOrder;)Ljava/nio/ByteBuffer;");
    byte_order_native = env->GetStaticMethodID(
        byte_order_class, "nativeOrder", "()Ljava/nio/ByteOrder;");

    return !env->ExceptionCheck() && tensor_new != nullptr &&
        tensor_dtype != nullptr && tensor_data_buffer != nullptr &&
        tensor_shape != nullptr && evalue_from_tensor != nullptr &&
        evalue_from_long != nullptr && evalue_from_double != nullptr &&
        evalue_from_bool != nullptr && evalue_from_string != nullptr &&
        evalue_to_tensor != nullptr && evalue_to_long != nullptr &&
        evalue_to_double != nullptr && evalue_to_bool != nullptr &&
        evalue_type_code != nullptr && byte_buffer_order != nullptr &&
        byte_order_native != nullptr;
  }

 private:
  static jclass find_global_class(JNIEnv* env, const char* name) {
    jclass local = env->FindClass(name);
    if (local == nullptr) {
      return nullptr;
    }
    auto global = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    return global;
  }
};

RawJniCache g_raw_jni;

void set_exception(JNIEnv* env, Error error, const std::string& message) {
  executorch::jni_helper::setExecutorchPendingException(
      env, static_cast<uint32_t>(error), message);
}

executorch::extension::ExecuTorchJni* module_from_handle(
    JNIEnv* env,
    jlong native_handle) {
  if (native_handle == 0) {
    set_exception(env, Error::InvalidState, "Module has been destroyed");
    return nullptr;
  }
  return reinterpret_cast<executorch::extension::ExecuTorchJni*>(native_handle);
}

bool to_string(JNIEnv* env, jstring value, std::string& result) {
  if (value == nullptr) {
    set_exception(env, Error::InvalidArgument, "String argument is null");
    return false;
  }
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) {
    return false;
  }
  try {
    result.assign(chars);
  } catch (...) {
    env->ReleaseStringUTFChars(value, chars);
    throw;
  }
  env->ReleaseStringUTFChars(value, chars);
  return !env->ExceptionCheck();
}

bool to_string_vector(
    JNIEnv* env,
    jobjectArray values,
    std::vector<std::string>& result) {
  if (values == nullptr) {
    set_exception(env, Error::InvalidArgument, "String array is null");
    return false;
  }
  const jsize size = env->GetArrayLength(values);
  result.reserve(size);
  for (jsize i = 0; i < size; ++i) {
    auto value = static_cast<jstring>(env->GetObjectArrayElement(values, i));
    if (value == nullptr || env->ExceptionCheck()) {
      if (value != nullptr) {
        env->DeleteLocalRef(value);
      } else if (!env->ExceptionCheck()) {
        set_exception(
            env, Error::InvalidArgument, "String array contains null");
      }
      return false;
    }
    std::string converted;
    const bool converted_ok = to_string(env, value, converted);
    env->DeleteLocalRef(value);
    if (!converted_ok) {
      return false;
    }
    result.push_back(std::move(converted));
  }
  return true;
}

bool to_int_vector(JNIEnv* env, jintArray values, std::vector<jint>& result) {
  if (values == nullptr) {
    set_exception(env, Error::InvalidArgument, "Int array is null");
    return false;
  }
  const jsize size = env->GetArrayLength(values);
  result.resize(size);
  env->GetIntArrayRegion(values, 0, size, result.data());
  return !env->ExceptionCheck();
}

template <typename Container>
jobjectArray new_string_array(JNIEnv* env, const Container& values) {
  auto result = env->NewObjectArray(
      static_cast<jsize>(values.size()), g_raw_jni.string_class, nullptr);
  if (result == nullptr) {
    return nullptr;
  }
  jsize i = 0;
  for (const auto& item : values) {
    jstring value = env->NewStringUTF(item.c_str());
    if (value == nullptr) {
      env->DeleteLocalRef(result);
      return nullptr;
    }
    env->SetObjectArrayElement(result, i, value);
    env->DeleteLocalRef(value);
    if (env->ExceptionCheck()) {
      env->DeleteLocalRef(result);
      return nullptr;
    }
    ++i;
  }
  return result;
}

} // namespace

namespace {

jobject new_jtensor_from_tensor(
    JNIEnv* env,
    const executorch::aten::Tensor& tensor) {
  const auto scalar_type = tensor.scalar_type();
  if (scalar_type_to_java_dtype.count(scalar_type) == 0) {
    std::stringstream message;
    message << "executorch::aten::Tensor scalar type "
            << static_cast<int>(scalar_type)
            << " is not supported on java side";
    set_exception(env, Error::InvalidArgument, message.str());
    return nullptr;
  }

  const auto sizes = tensor.sizes();
  jlongArray shape = env->NewLongArray(static_cast<jsize>(sizes.size()));
  if (shape == nullptr) {
    return nullptr;
  }
  std::vector<jlong> shape_values(sizes.begin(), sizes.end());
  env->SetLongArrayRegion(
      shape, 0, static_cast<jsize>(shape_values.size()), shape_values.data());
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(shape);
    return nullptr;
  }

  jobject buffer = env->NewDirectByteBuffer(
      const_cast<void*>(tensor.const_data_ptr()),
      static_cast<jlong>(tensor.nbytes()));
  if (buffer == nullptr) {
    env->DeleteLocalRef(shape);
    return nullptr;
  }

  jobject native_order = env->CallStaticObjectMethod(
      g_raw_jni.byte_order_class, g_raw_jni.byte_order_native);
  if (native_order == nullptr || env->ExceptionCheck()) {
    if (native_order != nullptr) {
      env->DeleteLocalRef(native_order);
    }
    env->DeleteLocalRef(buffer);
    env->DeleteLocalRef(shape);
    return nullptr;
  }
  jobject ordered_buffer =
      env->CallObjectMethod(buffer, g_raw_jni.byte_buffer_order, native_order);
  env->DeleteLocalRef(native_order);
  if (ordered_buffer != nullptr) {
    env->DeleteLocalRef(ordered_buffer);
  }
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(buffer);
    env->DeleteLocalRef(shape);
    return nullptr;
  }

  jobject result = env->CallStaticObjectMethod(
      g_raw_jni.tensor_class,
      g_raw_jni.tensor_new,
      buffer,
      shape,
      static_cast<jint>(scalar_type_to_java_dtype.at(scalar_type)));
  env->DeleteLocalRef(buffer);
  env->DeleteLocalRef(shape);
  return env->ExceptionCheck() ? nullptr : result;
}

TensorPtr new_tensor_from_jtensor(JNIEnv* env, jobject tensor) {
  if (tensor == nullptr) {
    set_exception(env, Error::InvalidArgument, "Tensor is null");
    return nullptr;
  }

  const jint dtype = env->CallIntMethod(tensor, g_raw_jni.tensor_dtype);
  if (env->ExceptionCheck()) {
    return nullptr;
  }
  auto shape = static_cast<jlongArray>(
      env->GetObjectField(tensor, g_raw_jni.tensor_shape));
  if (env->ExceptionCheck() || shape == nullptr) {
    if (!env->ExceptionCheck()) {
      set_exception(env, Error::InvalidArgument, "Tensor shape is null");
    }
    return nullptr;
  }
  jobject buffer = env->CallObjectMethod(tensor, g_raw_jni.tensor_data_buffer);
  if (env->ExceptionCheck() || buffer == nullptr) {
    env->DeleteLocalRef(shape);
    if (buffer != nullptr) {
      env->DeleteLocalRef(buffer);
    } else if (!env->ExceptionCheck()) {
      set_exception(env, Error::InvalidArgument, "Tensor buffer is null");
    }
    return nullptr;
  }

  const jsize rank = env->GetArrayLength(shape);
  std::vector<jlong> java_sizes(rank);
  env->GetLongArrayRegion(shape, 0, rank, java_sizes.data());
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(shape);
    env->DeleteLocalRef(buffer);
    return nullptr;
  }

  std::vector<executorch::aten::SizesType> sizes;
  sizes.reserve(rank);
  int64_t numel = 1;
  for (const auto size : java_sizes) {
    sizes.push_back(size);
    numel *= size;
  }

  const auto scalar_type_it = java_dtype_to_scalar_type.find(dtype);
  if (scalar_type_it == java_dtype_to_scalar_type.end()) {
    std::stringstream message;
    message << "Unknown Tensor jdtype: [" << dtype << "]";
    set_exception(env, Error::InvalidArgument, message.str());
    env->DeleteLocalRef(shape);
    env->DeleteLocalRef(buffer);
    return nullptr;
  }

  const auto scalar_type = scalar_type_it->second;
  const jlong capacity = env->GetDirectBufferCapacity(buffer);
  const jlong expected_elements = static_cast<jlong>(numel);
  const jlong expected_bytes = expected_elements *
      static_cast<jlong>(executorch::runtime::elementSize(scalar_type));
  if (capacity < 0 ||
      (capacity != expected_elements && capacity != expected_bytes)) {
    std::stringstream message;
    message << "Tensor dimensions(elements number: " << numel
            << ") inconsistent with buffer capacity " << capacity
            << " (element size bytes: "
            << executorch::runtime::elementSize(scalar_type) << ")";
    set_exception(env, Error::InvalidArgument, message.str());
    env->DeleteLocalRef(shape);
    env->DeleteLocalRef(buffer);
    return nullptr;
  }

  void* data = env->GetDirectBufferAddress(buffer);
  if (data == nullptr && expected_bytes != 0) {
    set_exception(
        env,
        Error::InvalidArgument,
        "Tensor buffer is not direct or has invalid address");
    env->DeleteLocalRef(shape);
    env->DeleteLocalRef(buffer);
    return nullptr;
  }

  auto result = from_blob(data, sizes, scalar_type);
  env->DeleteLocalRef(shape);
  env->DeleteLocalRef(buffer);
  return result;
}

jobject new_jevalue_from_evalue(JNIEnv* env, EValue value) {
  if (value.isTensor()) {
    jobject tensor = new_jtensor_from_tensor(env, value.toTensor());
    if (tensor == nullptr) {
      return nullptr;
    }
    jobject result = env->CallStaticObjectMethod(
        g_raw_jni.evalue_class, g_raw_jni.evalue_from_tensor, tensor);
    env->DeleteLocalRef(tensor);
    return env->ExceptionCheck() ? nullptr : result;
  }
  if (value.isInt()) {
    return env->CallStaticObjectMethod(
        g_raw_jni.evalue_class,
        g_raw_jni.evalue_from_long,
        static_cast<jlong>(value.toInt()));
  }
  if (value.isDouble()) {
    return env->CallStaticObjectMethod(
        g_raw_jni.evalue_class,
        g_raw_jni.evalue_from_double,
        static_cast<jdouble>(value.toDouble()));
  }
  if (value.isBool()) {
    return env->CallStaticObjectMethod(
        g_raw_jni.evalue_class,
        g_raw_jni.evalue_from_bool,
        static_cast<jboolean>(value.toBool()));
  }
  if (value.isString()) {
    const std::string string_value(
        value.toString().begin(), value.toString().end());
    jstring string = env->NewStringUTF(string_value.c_str());
    if (string == nullptr) {
      return nullptr;
    }
    jobject result = env->CallStaticObjectMethod(
        g_raw_jni.evalue_class, g_raw_jni.evalue_from_string, string);
    env->DeleteLocalRef(string);
    return env->ExceptionCheck() ? nullptr : result;
  }

  std::stringstream message;
  message << "Unknown EValue type: [" << static_cast<int>(value.tag) << "]";
  set_exception(env, Error::InvalidArgument, message.str());
  return nullptr;
}

bool append_evalue(
    JNIEnv* env,
    jobject java_value,
    std::vector<TensorPtr>& tensors,
    std::vector<EValue>& values) {
  if (java_value == nullptr) {
    set_exception(env, Error::InvalidArgument, "Input EValue is null");
    return false;
  }

  const jint type_code =
      env->GetIntField(java_value, g_raw_jni.evalue_type_code);
  if (env->ExceptionCheck()) {
    return false;
  }
  if (type_code == kTypeCodeTensor) {
    jobject tensor =
        env->CallObjectMethod(java_value, g_raw_jni.evalue_to_tensor);
    if (tensor == nullptr || env->ExceptionCheck()) {
      if (tensor != nullptr) {
        env->DeleteLocalRef(tensor);
      }
      return false;
    }
    auto native_tensor = new_tensor_from_jtensor(env, tensor);
    env->DeleteLocalRef(tensor);
    if (native_tensor == nullptr) {
      return false;
    }
    tensors.push_back(std::move(native_tensor));
    values.emplace_back(tensors.back());
    return true;
  }
  if (type_code == kTypeCodeInt) {
    const jlong value =
        env->CallLongMethod(java_value, g_raw_jni.evalue_to_long);
    if (!env->ExceptionCheck()) {
      values.emplace_back(static_cast<int64_t>(value));
      return true;
    }
    return false;
  }
  if (type_code == kTypeCodeDouble) {
    const jdouble value =
        env->CallDoubleMethod(java_value, g_raw_jni.evalue_to_double);
    if (!env->ExceptionCheck()) {
      values.emplace_back(static_cast<double>(value));
      return true;
    }
    return false;
  }
  if (type_code == kTypeCodeBool) {
    const jboolean value =
        env->CallBooleanMethod(java_value, g_raw_jni.evalue_to_bool);
    if (!env->ExceptionCheck()) {
      values.emplace_back(static_cast<bool>(value));
      return true;
    }
    return false;
  }

  std::stringstream message;
  message << "Unsupported input EValue type code: " << type_code;
  set_exception(env, Error::InvalidArgument, message.str());
  return false;
}

template <typename Container>
jobjectArray new_evalue_array(JNIEnv* env, const Container& values) {
  auto result = env->NewObjectArray(
      static_cast<jsize>(values.size()), g_raw_jni.evalue_class, nullptr);
  if (result == nullptr) {
    return nullptr;
  }
  for (jsize i = 0; i < static_cast<jsize>(values.size()); ++i) {
    jobject value = new_jevalue_from_evalue(env, values[i]);
    if (value == nullptr) {
      env->DeleteLocalRef(result);
      return nullptr;
    }
    env->SetObjectArrayElement(result, i, value);
    env->DeleteLocalRef(value);
    if (env->ExceptionCheck()) {
      env->DeleteLocalRef(result);
      return nullptr;
    }
  }
  return result;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_Module_nativeCreate(
    JNIEnv* env,
    jclass,
    jstring model_path,
    jint load_mode,
    jint num_threads) {
  try {
    std::string path;
    if (!to_string(env, model_path, path)) {
      return 0;
    }
    auto module = std::make_unique<executorch::extension::ExecuTorchJni>(
        path, load_mode, num_threads);
    return reinterpret_cast<jlong>(module.release());
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to create Module: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to create Module");
  }
  return 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_Module_nativeCreateWithOptions(
    JNIEnv* env,
    jclass,
    jstring model_path,
    jint load_mode,
    jint num_threads,
    jobjectArray backend_names,
    jobjectArray option_keys,
    jintArray option_values) {
  try {
    std::string path;
    std::vector<std::string> backend_name_values;
    std::vector<std::string> option_key_values;
    std::vector<jint> option_value_values;
    if (!to_string(env, model_path, path) ||
        !to_string_vector(env, backend_names, backend_name_values) ||
        !to_string_vector(env, option_keys, option_key_values) ||
        !to_int_vector(env, option_values, option_value_values)) {
      return 0;
    }

    auto module = std::make_unique<executorch::extension::ExecuTorchJni>(
        path, load_mode, num_threads);
    std::string error_message;
    const auto error = module->load_with_options(
        backend_name_values,
        option_key_values,
        option_value_values,
        error_message);
    if (error != Error::Ok) {
      set_exception(env, error, error_message);
      return 0;
    }
    return reinterpret_cast<jlong>(module.release());
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to create Module: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to create Module");
  }
  return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_org_pytorch_executorch_Module_nativeDestroy(
    JNIEnv*,
    jclass,
    jlong native_handle) {
  delete reinterpret_cast<executorch::extension::ExecuTorchJni*>(native_handle);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeExecute(
    JNIEnv* env,
    jclass,
    jlong native_handle,
    jstring method_name,
    jobjectArray inputs) {
  auto* native = module_from_handle(env, native_handle);
  if (native == nullptr) {
    return nullptr;
  }

  try {
    std::string method;
    if (!to_string(env, method_name, method)) {
      return nullptr;
    }
    const jsize input_count =
        inputs == nullptr ? 0 : env->GetArrayLength(inputs);

    if (input_count == 0) {
      auto error = native->module()->load_method(method);
      if (error != Error::Ok) {
        std::stringstream message;
        message << "Cannot get method names [Native Error: 0x" << std::hex
                << std::uppercase << static_cast<uint32_t>(error) << "]";
        set_exception(env, error, message.str());
        return nullptr;
      }
      auto* underlying_method = native->get_method(method);
      if (underlying_method == nullptr) {
        set_exception(
            env, Error::InvalidArgument, "Method not found: " + method);
        return nullptr;
      }
      [[maybe_unused]] auto input_tensors =
          prepare_input_tensors(*underlying_method);
      error = underlying_method->execute();
      if (error != Error::Ok) {
        set_exception(env, error, "Execution failed for method: " + method);
        return nullptr;
      }

      auto result = env->NewObjectArray(
          static_cast<jsize>(underlying_method->outputs_size()),
          g_raw_jni.evalue_class,
          nullptr);
      if (result == nullptr) {
        return nullptr;
      }
      for (size_t i = 0; i < underlying_method->outputs_size(); ++i) {
        jobject value =
            new_jevalue_from_evalue(env, underlying_method->get_output(i));
        if (value == nullptr) {
          env->DeleteLocalRef(result);
          return nullptr;
        }
        env->SetObjectArrayElement(result, static_cast<jsize>(i), value);
        env->DeleteLocalRef(value);
        if (env->ExceptionCheck()) {
          env->DeleteLocalRef(result);
          return nullptr;
        }
      }
      return result;
    }

    std::vector<EValue> values;
    std::vector<TensorPtr> tensors;
    values.reserve(input_count);
    tensors.reserve(input_count);
    for (jsize i = 0; i < input_count; ++i) {
      jobject value = env->GetObjectArrayElement(inputs, i);
      if (env->ExceptionCheck() ||
          !append_evalue(env, value, tensors, values)) {
        if (value != nullptr) {
          env->DeleteLocalRef(value);
        }
        return nullptr;
      }
      env->DeleteLocalRef(value);
    }

#if defined(ET_USE_THREADPOOL) && \
    defined(EXECUTORCH_HAS_THREADPOOL_USE_N_THREADS_GUARD)
    ::executorch::extension::threadpool::UseNThreadsThreadPoolGuard
        thread_pool_guard(native->num_threads());
#endif

#ifdef EXECUTORCH_ANDROID_PROFILING
    const auto start = std::chrono::high_resolution_clock::now();
    auto result = native->module()->execute(method, values);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    ET_LOG(Debug, "Execution time: %lld ms.", duration);
#else
    auto result = native->module()->execute(method, values);
#endif

    if (!result.ok()) {
      set_exception(
          env, result.error(), "Execution failed for method: " + method);
      return nullptr;
    }
    return new_evalue_array(env, result.get());
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Module execution failed: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Module execution failed");
  }
  return nullptr;
}

extern "C" JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_Module_nativeLoadMethod(
    JNIEnv* env,
    jclass,
    jlong native_handle,
    jstring method_name) {
  auto* native = module_from_handle(env, native_handle);
  if (native == nullptr) {
    return static_cast<jint>(Error::InvalidState);
  }
  try {
    std::string method;
    if (!to_string(env, method_name, method)) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    return static_cast<jint>(native->module()->load_method(method));
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to load Module method: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to load Module method");
  }
  return static_cast<jint>(Error::Internal);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeGetMethods(
    JNIEnv* env,
    jclass,
    jlong native_handle) {
  auto* native = module_from_handle(env, native_handle);
  if (native == nullptr) {
    return nullptr;
  }
  try {
    auto names = native->module()->method_names();
    if (!names.ok()) {
      std::stringstream message;
      message << "Cannot get load module [Native Error: 0x" << std::hex
              << std::uppercase << static_cast<uint32_t>(names.error()) << "]";
      set_exception(env, Error::InvalidArgument, message.str());
      return nullptr;
    }
    return new_string_array(env, names.get());
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to get Module methods: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to get Module methods");
  }
  return nullptr;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeGetUsedBackends(
    JNIEnv* env,
    jclass,
    jlong native_handle,
    jstring method_name) {
  auto* native = module_from_handle(env, native_handle);
  if (native == nullptr) {
    return nullptr;
  }
  try {
    std::string method;
    if (!to_string(env, method_name, method)) {
      return nullptr;
    }

    auto metadata = native->module()->method_meta(method);
    if (!metadata.ok()) {
      std::stringstream message;
      message << "Cannot get method meta for '" << method
              << "' [Native Error: 0x" << std::hex << std::uppercase
              << static_cast<uint32_t>(metadata.error()) << "]";
      set_exception(env, metadata.error(), message.str());
      return nullptr;
    }

    std::unordered_set<std::string> unique_backends;
    auto method_metadata = metadata.get();
    for (size_t i = 0; i < method_metadata.num_backends(); ++i) {
      auto backend = method_metadata.get_backend_name(i);
      if (backend.ok()) {
        unique_backends.insert(backend.get());
      }
    }
    std::vector<std::string> backends(
        unique_backends.begin(), unique_backends.end());
    return new_string_array(env, backends);
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to get Module backends: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to get Module backends");
  }
  return nullptr;
}

namespace {

jobjectArray read_log_buffer(JNIEnv* env) {
  try {
    std::vector<std::string> entries;
#ifdef __ANDROID__
    access_log_buffer([&entries](std::vector<log_entry>& buffer) {
      entries.reserve(buffer.size());
      for (const auto& entry : buffer) {
        std::stringstream message;
        message << "[" << entry.timestamp << " " << entry.function << " "
                << entry.filename << ":" << entry.line << "] "
                << static_cast<char>(entry.level) << " " << entry.message;
        entries.push_back(message.str());
      }
    });
#endif
    return new_string_array(env, entries);
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to read Module log buffer: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to read Module log buffer");
  }
  return nullptr;
}

} // namespace

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeReadLogBuffer(
    JNIEnv* env,
    jclass,
    jlong native_handle) {
  if (module_from_handle(env, native_handle) == nullptr) {
    return nullptr;
  }
  return read_log_buffer(env);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeReadLogBufferStatic(
    JNIEnv* env,
    jclass) {
  return read_log_buffer(env);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_pytorch_executorch_Module_nativeEtdump(
    JNIEnv* env,
    jclass,
    jlong native_handle) {
  auto* native = module_from_handle(env, native_handle);
  if (native == nullptr) {
    return JNI_FALSE;
  }
  try {
    return native->etdump();
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to write ETDump: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to write ETDump");
  }
  return JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_pytorch_executorch_Module_nativeEtdumpTo(
    JNIEnv* env,
    jclass,
    jlong native_handle,
    jstring output_path) {
  auto* native = module_from_handle(env, native_handle);
  if (native == nullptr) {
    return JNI_FALSE;
  }
  try {
    std::string path;
    if (!to_string(env, output_path, path)) {
      return JNI_FALSE;
    }
    return native->etdump_to(path);
  } catch (const std::exception& error) {
    set_exception(
        env,
        Error::Internal,
        std::string("Failed to write ETDump: ") + error.what());
  } catch (...) {
    set_exception(env, Error::Internal, "Failed to write ETDump");
  }
  return JNI_FALSE;
}

#ifdef EXECUTORCH_BUILD_LLAMA_JNI
extern void register_natives_for_llm();
#else
// No op if we don't build LLM
void register_natives_for_llm() {}
#endif

#ifdef EXECUTORCH_BUILD_EXTENSION_TRAINING
extern void register_natives_for_training();
#else
// No op if we don't build training JNI
void register_natives_for_training() {}
#endif

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK ||
      !g_raw_jni.initialize(env)) {
    return JNI_ERR;
  }
  return facebook::jni::initialize(vm, [] {
    register_natives_for_llm();
    register_natives_for_training();
  });
}
