// AGXMTLCounterSampleBuffer: the A10X driver's counter sample buffer (agxa_mtl_counters.mm on 21H461), whose 8-byte
// samples live in an AGXBuffer the AGX kernel extension counts as counter sample storage.

#import "WebMetalCounterSampleBuffer.h"

#import "WebMetalBuffer.h"

#include <dlfcn.h>

namespace webmetal {

namespace {

// AGXMTLCounterSampleBuffer's refusals: an NSError of MTLCounterErrorDomain with the description as
// NSLocalizedDescriptionKey (none when the description is nil), autoreleased.
NSError *counterError(NSInteger code, NSString *description)
{
    NSDictionary *info = description ? [NSDictionary dictionaryWithObject:description forKey:NSLocalizedDescriptionKey]
                                     : nil;
    return [[NSError alloc] initWithDomain:MTLCounterErrorDomain code:code userInfo:info];
}

// A string constant Metal.framework exports without a declaration in the SDK.
NSString *metalString(id self, SEL _cmd, const char *symbol)
{
    auto address = static_cast<NSString *const *>(dlsym(RTLD_DEFAULT, symbol));
    if (!address)
        WMLoudMethod(self, _cmd, "the running Metal.framework does not export the counter names the driver uses");
    return *address;
}

}  // namespace

// validateSampleIndex (AGXMetalA10 0x21c9ced9c): MTLCounterDontSample (~0) samples nothing; an index below the
// buffer's sampleCount is a sample; any other is logged (agxa_mtl_counters.mm line 379) and becomes ~0.
bool ValidateSampleIndex(id<MTLCounterSampleBuffer> buffer, uint32_t *index)
{
    const uint32_t value = *index;
    if (value == UINT32_MAX)
        return false;
    const NSUInteger count = [buffer sampleCount];
    if (value < uint32_t(count))
        return true;
    WM_AGX_LOG_ARGS("agxa_mtl_counters.mm", 379, "validateSampleIndex", "Invalid sample index: %u > %u", value,
                    uint32_t(count));
    *index = UINT32_MAX;
    return false;
}

// setCounterSampleBuffer (AGXMetalA10 0x21c9ceaf4): an AGXMTLCounterSampleBuffer's sample storage becomes the slot's;
// a buffer of another class (line 292) or without storage (line 299) is logged and leaves the slot as it was.
bool AttachSampleStorage(__strong id *slot, id<MTLCounterSampleBuffer> buffer)
{
    if (![(id)buffer isKindOfClass:[AGXMTLCounterSampleBuffer class]]) {
        WM_AGX_LOG("agxa_mtl_counters.mm", 292, "setCounterSampleBuffer",
                   "Error, counter sample buffer is not of type AGXMTLCounterSampleBuffer");
        return false;
    }
    AGXBuffer *storage = [(AGXMTLCounterSampleBuffer *)buffer buffer];
    if (!storage) {
        WM_AGX_LOG("agxa_mtl_counters.mm", 299, "setCounterSampleBuffer", "Error, invalid counter sample storage");
        return false;
    }
    *slot = storage;
    return true;
}

}  // namespace webmetal

@implementation AGXMTLCounterSampleBuffer {
    // The driver's ivars, in its order: the device is not retained.
    __unsafe_unretained id<MTLDevice> _device;
    NSString *_label;
    NSUInteger _sampleCount;
    AGXBuffer *_buffer;
}

// -[AGXMTLCounterSampleBuffer initWithDevice:descriptor:error:] (AGXMetalA10 0x21c9ce600): clears *error and refuses a
// counter set that does not hold exactly one counter, one not named MTLCommonCounterSetTimestamp and a counter not
// named MTLCommonCounterTimestamp (MTLCounterSampleBufferErrorInvalid with the driver's descriptions); otherwise the
// length form with sampleCount 8-byte samples, the descriptor's label and its storage mode as resource options.
- (instancetype)initWithDevice:(id<MTLDevice>)device
                    descriptor:(MTLCounterSampleBufferDescriptor *)descriptor
                         error:(NSError *__autoreleasing *)error
{
    using namespace webmetal;
    if (error)
        *error = nil;
    NSString *refusal = nil;
    if (descriptor.counterSet.counters.count != 1) {
        refusal = @"Invalid counter set size. Expecting 1 counter";
    } else {
        NSString *foundSet = descriptor.counterSet.name;
        NSString *expectedSet = metalString(self, _cmd, "MTLCommonCounterSetTimestamp");
        if (![foundSet isEqualToString:expectedSet]) {
            refusal = [NSString stringWithFormat:@"Invalid counter set name. Expecting: %@. Found: %@", expectedSet,
                                                 foundSet];
        } else {
            NSString *foundCounter = descriptor.counterSet.counters[0].name;
            NSString *expectedCounter = metalString(self, _cmd, "MTLCommonCounterTimestamp");
            if (![foundCounter isEqualToString:expectedCounter])
                refusal = [NSString stringWithFormat:@"Invalid counter name. Expecting: %@. Found: %@",
                                                     expectedCounter, foundCounter];
            else
                return [self initWithDevice:device
                                     length:uint64_t(descriptor.sampleCount) << 3
                                      label:descriptor.label
                                    options:(uint64_t(descriptor.storageMode) << 4) & 0xf0
                                      error:error];
        }
    }
    if (error)
        *error = counterError(MTLCounterSampleBufferErrorInvalid, refusal);
    return nil;
}

// -[AGXMTLCounterSampleBuffer initWithDevice:length:label:options:error:] (AGXMetalA10 0x21c9ce7d0): clears *error;
// "Internal error" (MTLCounterSampleBufferErrorInternal) when NSObject's initializer fails; refuses a length outside 1 to
// 32768 bytes and one that is not a multiple of 8 (MTLCounterSampleBufferErrorInvalid, 0x21c9ce828 to 0x21c9ce900);
// then the sample storage, -[AGXBuffer initCounterSampleBufferWithDevice:length:options:] (0x21c8a4544), whose
// failure is "Cannot allocate sample buffer" (MTLCounterSampleBufferErrorOutOfMemory). It keeps the device unretained,
// a copy of the label and length / 8 samples.
- (instancetype)initWithDevice:(id<MTLDevice>)device
                        length:(NSUInteger)length
                         label:(NSString *)label
                       options:(MTLResourceOptions)options
                         error:(NSError *__autoreleasing *)error
{
    using namespace webmetal;
    if (error)
        *error = nil;
    self = [super init];
    if (!self) {
        if (error)
            *error = counterError(MTLCounterSampleBufferErrorInternal, @"Internal error");
        return nil;
    }
    if (length - 1 >= 0x8000 || (length & 7)) {
        NSString *refusal =
            length - 1 >= 0x8000
                ? [NSString stringWithFormat:@"Invalid sample buffer length: %llu B. Expected range: %zu -> %u",
                                             static_cast<unsigned long long>(length), size_t(8), 32768u]
                : [NSString stringWithFormat:@"Invalid sample buffer length: %llu B. Length must be aligned to %zu B",
                                             static_cast<unsigned long long>(length), size_t(8)];
        if (error)
            *error = counterError(MTLCounterSampleBufferErrorInvalid, refusal);
        return nil;
    }
    _buffer = NewCounterSampleStorage(static_cast<AGXA10FamilyDevice *>(device), _cmd, length, options);
    if (!_buffer) {
        if (error)
            *error = counterError(MTLCounterSampleBufferErrorOutOfMemory, @"Cannot allocate sample buffer");
        return nil;
    }
    _device = device;
    _label = [label copy];
    _sampleCount = length >> 3;
    return self;
}

// -[AGXMTLCounterSampleBuffer device], -label, -sampleCount, -buffer (AGXMetalA10 0x21c9cea3c to 0x21c9cea5c): the
// ivars; -sampleSize (0x21c9cea54): the constant 8.
- (id<MTLDevice>)device
{
    return _device;
}

- (NSString *)label
{
    return _label;
}

- (NSUInteger)sampleCount
{
    return _sampleCount;
}

- (NSUInteger)sampleSize
{
    return 8;
}

- (AGXBuffer *)buffer
{
    return _buffer;
}

// -[AGXMTLCounterSampleBuffer resolveCounterRange:] (AGXMetalA10 0x21c9cea64): nil for an empty range and one past
// sampleCount (location + length in 64 bits); otherwise a new NSMutableData of the range's samples, copied from the
// storage's contents.
- (NSData *)resolveCounterRange:(NSRange)range
{
    if (!range.length || range.location + range.length > _sampleCount)
        return nil;
    NSMutableData *data = [NSMutableData dataWithLength:range.length << 3];
    if (data)
        memmove(data.mutableBytes, static_cast<uint8_t *>([(id<MTLBuffer>)_buffer contents]) + (range.location << 3),
                range.length << 3);
    return data;
}

@end
