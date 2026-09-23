/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import java.lang.ref.PhantomReference
import java.lang.ref.ReferenceQueue
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

internal object NativeHandleCleaner {
  internal fun interface Destructor {
    fun destroy(nativeHandle: Long)
  }

  internal interface Cleanable {
    fun clean()
  }

  private val referenceQueue = ReferenceQueue<Any>()
  private val liveReferences =
      Collections.newSetFromMap(ConcurrentHashMap<CleanableReference, Boolean>())

  init {
    Thread(
            {
              while (true) {
                try {
                  (referenceQueue.remove() as CleanableReference).clean()
                } catch (_: InterruptedException) {
                  // Keep processing cleanup work after a spurious interruption.
                } catch (_: Throwable) {
                  // A failed destructor must not stop cleanup for other handles.
                }
              }
            },
            "ExecuTorch native cleanup",
        )
        .apply {
          isDaemon = true
          start()
        }
  }

  fun register(owner: Any, nativeHandle: AtomicLong, destructor: Destructor): Cleanable =
      CleanableReference(owner, nativeHandle, destructor).also { liveReferences.add(it) }

  private class CleanableReference(
      owner: Any,
      private val nativeHandle: AtomicLong,
      private val destructor: Destructor,
  ) : PhantomReference<Any>(owner, referenceQueue), Cleanable {
    override fun clean() {
      if (!liveReferences.remove(this)) {
        return
      }
      val handle = nativeHandle.getAndSet(0L)
      try {
        if (handle != 0L) {
          destructor.destroy(handle)
        }
      } finally {
        clear()
      }
    }
  }
}
