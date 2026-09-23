/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import com.facebook.jni.annotations.DoNotStrip
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import java.io.Closeable
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import org.pytorch.executorch.annotations.Experimental

/**
 * Java wrapper for ExecuTorch Module.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
open class Module
private constructor(
    moduleAbsolutePath: String,
    loadMode: Int,
    numThreads: Int,
    backendOptions: BackendOptionsMap?,
) : Closeable {

  private val mNativeHandle = AtomicLong(0L)
  private val mCleanup = NativeHandleCleaner.register(this, mNativeHandle, MODULE_DESTRUCTOR)
  private val mMethodMetadata: Map<String, MethodMetadata>

  /** Lock protecting the non-thread-safe native module. */
  private val mLock = ReentrantLock()

  init {
    ExecuTorchRuntime.getRuntime()
    val handle =
        if (backendOptions == null || backendOptions.isEmpty()) {
          nativeCreate(moduleAbsolutePath, loadMode, numThreads)
        } else {
          val (backendNames, optionKeys, optionValues) = backendOptions.toJniArrays()
          nativeCreateWithOptions(
              moduleAbsolutePath,
              loadMode,
              numThreads,
              backendNames,
              optionKeys,
              optionValues,
          )
        }
    check(handle != 0L) { "Failed to create native Module" }
    mNativeHandle.set(handle)
    try {
      mMethodMetadata = populateMethodMeta(handle)
    } catch (throwable: Throwable) {
      mCleanup.clean()
      throw throwable
    }
  }

  private fun populateMethodMeta(handle: Long): Map<String, MethodMetadata> {
    val methods = nativeGetMethods(handle)
    val metadata = HashMap<String, MethodMetadata>()
    for (name in methods) {
      metadata[name] = MethodMetadata(name, nativeGetUsedBackends(handle, name))
    }
    return metadata
  }

  private fun requireNativeHandle(): Long {
    val handle = mNativeHandle.get()
    check(handle != 0L) { "Module has been destroyed" }
    return handle
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the ExecuTorch module's 'forward' method. Note: if method 'forward'
   *   requires inputs but no inputs are given, the function will not error out, but run 'forward'
   *   with sample inputs.
   * @return return value from the 'forward' method.
   */
  open fun forward(vararg inputs: EValue): Array<EValue> = execute("forward", *inputs)

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the ExecuTorch method to run.
   * @param inputs arguments that will be passed to ExecuTorch method.
   * @return return value from the method.
   */
  open fun execute(methodName: String, vararg inputs: EValue): Array<EValue> {
    mLock.lock()
    try {
      return nativeExecute(requireNativeHandle(), methodName, inputs)
    } finally {
      mLock.unlock()
    }
  }

  /**
   * Load a method on this module. This might help with the first time inference performance,
   * because otherwise the method is loaded lazily when it's execute. Note: this function is
   * synchronous, and will block until the method is loaded. Therefore, it is recommended to call
   * this on a background thread. However, users need to make sure that they don't execute before
   * this function returns.
   */
  open fun loadMethod(methodName: String) {
    mLock.lock()
    try {
      val errorCode = nativeLoadMethod(requireNativeHandle(), methodName)
      if (errorCode != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(
            errorCode,
            "Failed to load method: $methodName",
        )
      }
    } finally {
      mLock.unlock()
    }
  }

  /**
   * Returns the names of methods.
   *
   * @return name of methods in this Module
   */
  open fun getMethods(): Array<String> {
    mLock.lock()
    try {
      return nativeGetMethods(requireNativeHandle())
    } finally {
      mLock.unlock()
    }
  }

  /**
   * Get the corresponding [MethodMetadata] for a method
   *
   * @param name method name
   * @return [MethodMetadata] for this method
   */
  open fun getMethodMetadata(name: String): MethodMetadata {
    mLock.lock()
    try {
      requireNativeHandle()
      return mMethodMetadata[name]
          ?: throw IllegalArgumentException("method $name does not exist for this module")
    } finally {
      mLock.unlock()
    }
  }

  /** Retrieve the in-memory log buffer, containing the most recent ExecuTorch log entries. */
  open fun readLogBuffer(): Array<String>? {
    mLock.lock()
    try {
      return nativeReadLogBuffer(requireNativeHandle())
    } finally {
      mLock.unlock()
    }
  }

  /**
   * Dump the ExecuTorch ETRecord file to /data/local/tmp/result.etdump.
   *
   * Currently for internal (minibench) use only.
   *
   * @return true if the etdump was successfully written, false otherwise.
   */
  @Experimental
  open fun etdump(): Boolean {
    mLock.lock()
    try {
      return nativeEtdump(requireNativeHandle())
    } finally {
      mLock.unlock()
    }
  }

  /**
   * Dump the ExecuTorch ETDump file to [outputPath].
   *
   * @param outputPath absolute path to write the etdump file to.
   * @return true if the etdump was successfully written, false otherwise.
   */
  @Experimental
  open fun etdump(outputPath: String): Boolean {
    mLock.lock()
    try {
      return nativeEtdumpTo(requireNativeHandle(), outputPath)
    } finally {
      mLock.unlock()
    }
  }

  /**
   * Explicitly destroys the native Module object. The object is also released after this [Module]
   * becomes unreachable, but calling this method is recommended because the timing of garbage
   * collection is not guaranteed.
   */
  open fun destroy() {
    if (mLock.tryLock()) {
      try {
        mCleanup.clean()
      } finally {
        mLock.unlock()
      }
    } else {
      throw IllegalStateException("Cannot destroy module while method is executing")
    }
  }

  override fun close() {
    destroy()
  }

  companion object {
    private val MODULE_DESTRUCTOR = NativeHandleCleaner.Destructor(::nativeDestroy)

    init {
      if (!NativeLoader.isInitialized()) {
        NativeLoader.init(SystemDelegate())
      }
      NativeLoader.loadLibrary("executorch")
    }

    /** Load mode for the module. Load the whole file as a buffer. */
    const val LOAD_MODE_FILE = 0

    /** Load mode for the module. Use mmap to load pages into memory. */
    const val LOAD_MODE_MMAP = 1

    /** Load mode for the module. Use memory locking and handle errors. */
    const val LOAD_MODE_MMAP_USE_MLOCK = 2

    /** Load mode for the module. Use memory locking and ignore errors. */
    const val LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS = 3

    /**
     * Loads a serialized ExecuTorch module from the specified path on the disk.
     *
     * @param modelPath path to file that contains the serialized ExecuTorch module.
     * @param loadMode load mode for the module. See constants in [Module].
     * @param numThreads the number of threads to use for inference. A value of 0 defaults to a
     *   hardware-specific default.
     * @return new [Module] object which owns the model module.
     */
    @JvmStatic
    @JvmOverloads
    fun load(modelPath: String?, loadMode: Int = LOAD_MODE_FILE, numThreads: Int = 0): Module {
      ExecuTorchRuntime.validateFilePath(modelPath, "model path")
      return Module(modelPath!!, loadMode, numThreads, null)
    }

    /**
     * Loads a serialized ExecuTorch module and applies per-backend options at delegate-init (load)
     * time. Prefer this over a process-global setter: the options travel with the load, so there is
     * no ordering or thread-safety hazard to manage.
     *
     * @param modelPath path to file that contains the serialized ExecuTorch module.
     * @param options backend options to apply at load, e.g.
     *   `BackendOptionsMap().setInt("XnnpackBackend", "workspace_sharing_mode", 2)`.
     * @param loadMode load mode for the module. See constants in [Module].
     * @param numThreads the number of threads to use for inference. A value of 0 defaults to a
     *   hardware-specific default.
     * @return new [Module] object which owns the model module.
     */
    @JvmStatic
    @JvmOverloads
    fun load(
        modelPath: String?,
        options: BackendOptionsMap,
        loadMode: Int = LOAD_MODE_FILE,
        numThreads: Int = 0,
    ): Module {
      ExecuTorchRuntime.validateFilePath(modelPath, "model path")
      // Non-null already guaranteed by validateFilePath above.
      return Module(checkNotNull(modelPath), loadMode, numThreads, options)
    }

    @DoNotStrip
    @JvmStatic
    private external fun nativeCreate(
        moduleAbsolutePath: String,
        loadMode: Int,
        numThreads: Int,
    ): Long

    @DoNotStrip
    @JvmStatic
    private external fun nativeCreateWithOptions(
        moduleAbsolutePath: String,
        loadMode: Int,
        numThreads: Int,
        backendNames: Array<String>,
        optionKeys: Array<String>,
        optionValues: IntArray,
    ): Long

    @DoNotStrip @JvmStatic private external fun nativeDestroy(nativeHandle: Long)

    @DoNotStrip
    @JvmStatic
    private external fun nativeExecute(
        nativeHandle: Long,
        methodName: String,
        inputs: Array<out EValue>,
    ): Array<EValue>

    @DoNotStrip
    @JvmStatic
    private external fun nativeLoadMethod(nativeHandle: Long, methodName: String): Int

    @DoNotStrip @JvmStatic private external fun nativeGetMethods(nativeHandle: Long): Array<String>

    @DoNotStrip
    @JvmStatic
    private external fun nativeGetUsedBackends(
        nativeHandle: Long,
        methodName: String,
    ): Array<String>

    @DoNotStrip
    @JvmStatic
    private external fun nativeReadLogBuffer(nativeHandle: Long): Array<String>?

    @DoNotStrip @JvmStatic private external fun nativeEtdump(nativeHandle: Long): Boolean

    @DoNotStrip
    @JvmStatic
    private external fun nativeEtdumpTo(nativeHandle: Long, outputPath: String): Boolean

    @DoNotStrip @JvmStatic fun readLogBufferStatic(): Array<String>? = nativeReadLogBufferStatic()

    @DoNotStrip @JvmStatic private external fun nativeReadLogBufferStatic(): Array<String>?
  }
}
