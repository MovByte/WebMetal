// Counter sample buffers (WebMetalCounterSampleBuffer.mm): AGXMTLCounterSampleBuffer and its sample storage, as on
// 21H461.

#pragma once

#import "WebMetalResources.h"

@interface AGXMTLCounterSampleBuffer : NSObject <MTLCounterSampleBuffer>
- (instancetype)initWithDevice:(id<MTLDevice>)device
                    descriptor:(MTLCounterSampleBufferDescriptor *)descriptor
                         error:(NSError *__autoreleasing *)error;
- (instancetype)initWithDevice:(id<MTLDevice>)device
                        length:(NSUInteger)length
                         label:(NSString *)label
                       options:(MTLResourceOptions)options
                         error:(NSError *__autoreleasing *)error;
// The sample storage, and the bytes of one sample.
- (AGXBuffer *)buffer;
- (NSUInteger)sampleSize;
@end

namespace webmetal {

// -[AGXBuffer initCounterSampleBufferWithDevice:length:options:] (WebMetalBuffer.mm): the sample storage, or nil.
AGXBuffer *NewCounterSampleStorage(AGXA10FamilyDevice *device, SEL _cmd, NSUInteger length, MTLResourceOptions options);

// The driver's checks of a sample index and its attachment of a counter sample buffer's storage to an encoder
// (WebMetalCounterSampleBuffer.mm).
bool ValidateSampleIndex(id<MTLCounterSampleBuffer> buffer, uint32_t *index);
bool AttachSampleStorage(__strong id *slot, id<MTLCounterSampleBuffer> buffer);

}  // namespace webmetal
