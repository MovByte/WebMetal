// Heaps (WebMetalHeap, standing in for AGXA10FamilyHeap): what the device and the heap's buffers use.

#pragma once

#import "WebMetalResources.h"

#include <pthread.h>

// Metal.framework's range allocator, which IOGPUMetalHeap keeps for an automatic heap. The structure is
// IOGPUMetalHeap._allocator's type on 21H461 ({MTLRangeAllocator=^{MTLRangeAllocatorElement}IQQQ}); the functions are
// Metal.framework's exports, called with the arguments IOGPU passes.
struct MTLRangeAllocatorElement;
struct MTLRangeAllocator {
    MTLRangeAllocatorElement *elements;
    uint32_t numElements;
    uint64_t capacity;
    uint64_t capacityIncrement;
    uint64_t defaultAlignmentMask;
};
static_assert(sizeof(MTLRangeAllocator) == 40, "IOGPUMetalHeap._allocator is 40 bytes on 21H461");

extern "C" {
void MTLRangeAllocatorInit(MTLRangeAllocator *allocator, uint64_t lastOffset, uint64_t start, uint32_t capacity);
void MTLRangeAllocatorDestroy(MTLRangeAllocator *allocator);
bool MTLRangeAllocatorAllocate(MTLRangeAllocator *allocator, uint64_t length, uint64_t *offset, uint64_t alignment);
void MTLRangeAllocatorDeallocate(MTLRangeAllocator *allocator, uint64_t offset, uint64_t length);
uint64_t MTLRangeAllocatorGetFreeSize(MTLRangeAllocator *allocator);
uint64_t MTLRangeAllocatorGetMaxFreeSize(MTLRangeAllocator *allocator, uint32_t alignment);
}

namespace webmetal {

// -[AGXA10FamilyDevice newHeapWithDescriptor:] for an automatic or placement heap.
WebMetalHeap *NewHeap(WebMetalDevice *device, SEL _cmd, MTLHeapDescriptor *descriptor);
uint64_t HeapHandle(WebMetalHeap *heap);

// A buffer over [offset, offset + rangeLength) of `heap`, which the heap has granted; `contents` is the range's CPU
// address (null for a private heap). -[IOGPUMetalBuffer initWithHeap:resource:offset:length:] gives it the heap's
// options (`heapOptions`); `options` are the caller's, carried to the host as given.
WebMetalBuffer *NewHeapBuffer(WebMetalHeap *heap, SEL _cmd, NSUInteger length, MTLResourceOptions options,
                              uint64_t heapOptions, uint64_t offset, uint64_t rangeLength, void *contents,
                              bool placement);

}  // namespace webmetal
