/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import java.util.concurrent.atomic.AtomicLong
import org.junit.Assert.assertEquals
import org.junit.Test

class NativeHandleCleanerTest {
  @Test
  fun cleanDestroysHandleOnce() {
    val owner = Any()
    val handle = AtomicLong(42L)
    val destroyedHandle = AtomicLong(0L)
    val destroyCalls = AtomicLong(0L)
    val cleanable =
        NativeHandleCleaner.register(owner, handle) { nativeHandle ->
          destroyedHandle.set(nativeHandle)
          destroyCalls.incrementAndGet()
        }

    cleanable.clean()
    cleanable.clean()

    assertEquals(0L, handle.get())
    assertEquals(42L, destroyedHandle.get())
    assertEquals(1L, destroyCalls.get())
  }

  @Test
  fun cleanIgnoresEmptyHandle() {
    val owner = Any()
    val handle = AtomicLong(0L)
    val destroyCalls = AtomicLong(0L)
    val cleanable = NativeHandleCleaner.register(owner, handle) { destroyCalls.incrementAndGet() }

    cleanable.clean()

    assertEquals(0L, destroyCalls.get())
  }
}
