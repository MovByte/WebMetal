// The device's SPI answers (AGXA10XDevice and AGXA10FamilyDevice on 21H461).

#import "WebMetalSPI.h"

#import <objc/runtime.h>

#include <dlfcn.h>
#include <mach/mach_time.h>

// The class adopts the driver's protocols; these categories implement the driver selectors they declare.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

// Metal.framework's counter objects (MTLCounterInternal, MTLCounterSetInternal), which no header declares.
@protocol WebMetalCounterInternal
- (instancetype)initWithName:(NSString *)name description:(NSString *)description;
- (instancetype)initWithName:(NSString *)name description:(NSString *)description counters:(NSArray *)counters;
@end

namespace {

// The device's -disableComputeEncoderCoalescing flag, as an associated object (absent: NO).
const char kComputeCoalescingKey = 0;

// A string constant Metal.framework exports without a declaration in the SDK.
NSString *metalString(id self, SEL _cmd, const char *symbol)
{
    auto address = static_cast<NSString *const *>(dlsym(RTLD_DEFAULT, symbol));
    if (!address)
        WMLoudMethod(self, _cmd, "the running Metal.framework does not export the counter names the driver uses");
    return *address;
}

// AGXMTLCounterSampleBuffer's refusals: an NSError of MTLCounterErrorDomain with the description as
// NSLocalizedDescriptionKey (none when the description is nil), autoreleased.
NSError *counterError(NSInteger code, NSString *description)
{
    NSDictionary *info = description ? [NSDictionary dictionaryWithObject:description forKey:NSLocalizedDescriptionKey]
                                     : nil;
    return [[NSError alloc] initWithDomain:MTLCounterErrorDomain code:code userInfo:info];
}

const char kCounterSampleStorage[] =
    "AGXMTLCounterSampleBuffer keeps its samples in an AGXBuffer that -[AGXBuffer initCounterSampleBufferWithDevice:"
    "length:options:] (0x21c8a4544) creates unsub-allocated with the AGX resource argument 0x10 at +0x5c of the creation "
    "arguments, whose effect in the AGX kernel extension (and so the contents a resolve reads before any pass samples) is "
    "not read; the passes' sample buffer attachments then have the GPU write timestamps into it at stage boundaries, and "
    "the host writes no GPU timestamps";

// -[AGXMTLCounterSampleBuffer initWithDevice:length:label:options:error:] (AGXMetalA10 0x21c9ce7d0): clears *error,
// refuses a length outside 1 to 32768 bytes and one that is not a multiple of 8 (MTLCounterSampleBufferErrorInvalid
// with the driver's descriptions, 0x21c9ce828 to 0x21c9ce900), then makes the sample storage.
id newCounterSampleBuffer(id self, SEL _cmd, uint64_t length, NSError *__autoreleasing *error)
{
    if (error)
        *error = nil;
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
    WMLoudMethod(self, _cmd, kCounterSampleStorage);
}

}  // namespace

@implementation AGXA10FamilyDevice (WebMetalSPI)

// -[AGXA10FamilyDevice newCounterSampleBufferWithDescriptor:error:] (AGXMetalA10 0x21c8db494) through
// -[AGXMTLCounterSampleBuffer initWithDevice:descriptor:error:] (0x21c9ce600): clears *error and refuses a counter set
// that does not hold exactly one counter, one not named MTLCommonCounterSetTimestamp and a counter not named
// MTLCommonCounterTimestamp (MTLCounterSampleBufferErrorInvalid with the driver's descriptions); otherwise it makes a
// buffer of sampleCount 8-byte samples with the descriptor's label and storage mode.
- (id<MTLCounterSampleBuffer>)newCounterSampleBufferWithDescriptor:(MTLCounterSampleBufferDescriptor *)descriptor
                                                             error:(NSError *__autoreleasing *)error
{
    if (error)
        *error = nil;
    NSString *refusal = nil;
    if (descriptor.counterSet.counters.count != 1) {
        refusal = @"Invalid counter set size. Expecting 1 counter";
    } else {
        NSString *expectedSet = metalString(self, _cmd, "MTLCommonCounterSetTimestamp");
        NSString *foundSet = descriptor.counterSet.name;
        NSString *expectedCounter = metalString(self, _cmd, "MTLCommonCounterTimestamp");
        if (![foundSet isEqualToString:expectedSet]) {
            refusal = [NSString stringWithFormat:@"Invalid counter set name. Expecting: %@. Found: %@", expectedSet,
                                                 foundSet];
        } else {
            NSString *foundCounter = descriptor.counterSet.counters[0].name;
            if (![foundCounter isEqualToString:expectedCounter])
                refusal = [NSString stringWithFormat:@"Invalid counter name. Expecting: %@. Found: %@",
                                                     expectedCounter, foundCounter];
            else
                return newCounterSampleBuffer(self, _cmd, uint64_t(descriptor.sampleCount) << 3, error);
        }
    }
    if (error)
        *error = counterError(MTLCounterSampleBufferErrorInvalid, refusal);
    return nil;
}

// -[AGXA10FamilyDevice newCounterSampleBufferWithLength:label:error:] (AGXMetalA10 0x21c8db4e0): the length form of the
// initializer with options 0.
- (id)newCounterSampleBufferWithLength:(NSUInteger)length label:(NSString *)label error:(NSError *__autoreleasing *)error
{
    return newCounterSampleBuffer(self, _cmd, length, error);
}

// -[AGXA10FamilyDevice vendorName] (AGXMetalA10 0x21c8d4d24): the constant string "Apple".
- (NSString *)vendorName
{
    return @"Apple";
}

// -[AGXA10FamilyDevice familyName] (AGXMetalA10 0x21c8d4d30): the constant string "A10", which AGXA10XDevice
// overrides.
- (NSString *)familyName
{
    return @"A10";
}

// -[AGXA10FamilyDevice productName] (AGXMetalA10 0x21c8d4d3c): the constant empty string.
- (NSString *)productName
{
    return @"";
}

// -[AGXA10FamilyDevice newRenderPipelineStateWithTileDescriptor:options:reflection:error:] (AGXMetalA10 0x21c8d8c64):
// the A10 family has no tile functions (supportsTileShaders is NO); the driver stores a nil pipeline for its internal
// builds' pipeline list (-_storeRenderPipelineState: at 0x21c8d8580 keeps only pipelines with a label, so nil changes
// nothing) and answers nil, leaving *reflection and *error as they are.
- (id<MTLRenderPipelineState>)newRenderPipelineStateWithTileDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                                                               options:(MTLPipelineOption)options
                                                            reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                                                 error:(__autoreleasing NSError **)error
{
    return nil;
}

// -[AGXA10FamilyDevice newRenderPipelineStateWithTileDescriptor:options:completionHandler:] (AGXMetalA10 0x21c8d8c80)
// returns at once: the completion handler never runs.
- (void)newRenderPipelineStateWithTileDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                                         options:(MTLPipelineOption)options
                               completionHandler:(MTLNewRenderPipelineStateWithReflectionCompletionHandler)handler
{
}

// -[AGXA10FamilyDevice newRenderPipelineStateWithTileDescriptor:completionHandler:] (AGXMetalA10 0x21c8d8c60) returns
// at once as well.
- (void)newRenderPipelineStateWithTileDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                               completionHandler:(MTLNewRenderPipelineStateWithReflectionCompletionHandler)handler
{
}

// -[AGXA10FamilyDevice newVisibleFunctionTableWithDescriptor:] (AGXMetalA10 0x21c8db4d8) answers nil: the A10 family
// has no function pointers.
- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableWithDescriptor:(MTLVisibleFunctionTableDescriptor *)descriptor
{
    return nil;
}

// -[AGXA10FamilyDevice newIndirectComputeCommandEncoderWithBuffer:] (AGXMetalA10 0x21c8d63f0) answers nil.
- (id)newIndirectComputeCommandEncoderWithBuffer:(id<MTLBuffer>)buffer
{
    return nil;
}

// -[AGXA10FamilyDevice setIndirectArgumentBufferDecodingData:] (AGXMetalA10 0x21c8db1b4) returns at once.
- (void)setIndirectArgumentBufferDecodingData:(id)data
{
}

// -[IOGPUMetalDevice _addResource:] and -_removeResource: (IOGPU 0x1fcb2c6e8, 0x1fcb2cb7c) return at once.
- (void)_addResource:(id)resource
{
}

- (void)_removeResource:(id)resource
{
}

// -[AGXA10FamilyDevice counterSets] (AGXMetalA10 0x21c8db3d8): a new array of one Metal.framework counter set,
// MTLCommonCounterSetTimestamp described by MTLCounterSetDescriptionTimestamp, holding one counter,
// MTLCommonCounterTimestamp described by MTLCounterDescriptionTimestamp.
- (NSArray<id<MTLCounterSet>> *)counterSets
{
    Class counterClass = objc_getClass("MTLCounterInternal");
    Class setClass = objc_getClass("MTLCounterSetInternal");
    if (!counterClass || !setClass)
        WMLoudMethod(self, _cmd, "the running Metal.framework has no MTLCounterInternal or MTLCounterSetInternal");
    id counter = [(id<WebMetalCounterInternal>)[counterClass alloc]
        initWithName:metalString(self, _cmd, "MTLCommonCounterTimestamp")
         description:metalString(self, _cmd, "MTLCounterDescriptionTimestamp")];
    id set = [(id<WebMetalCounterInternal>)[setClass alloc]
        initWithName:metalString(self, _cmd, "MTLCommonCounterSetTimestamp")
         description:metalString(self, _cmd, "MTLCounterSetDescriptionTimestamp")
            counters:[NSArray arrayWithObjects:counter, nil]];
    return [NSArray arrayWithObjects:set, nil];
}

// -[AGXA10FamilyDevice counterSampleBufferLimits] (AGXMetalA10 0x21c8db530): one constant dictionary (0x22b7d86a8),
// the same object on every call, of int numbers: maxBufferSize 32768, maxDeviceBufferNum 32, maxGlobalBufferNum 1024.
- (NSDictionary *)counterSampleBufferLimits
{
    static NSDictionary *limits = [[NSDictionary alloc]
        initWithObjects:@[ [NSNumber numberWithInt:32768], [NSNumber numberWithInt:32], [NSNumber numberWithInt:1024] ]
                forKeys:@[ @"maxBufferSize", @"maxDeviceBufferNum", @"maxGlobalBufferNum" ]];
    return limits;
}

// -[AGXA10FamilyDevice sampleTimestamps:gpuTimestamp:] (AGXMetalA10 0x21c8db364): one mach_absolute_time reading
// converted to nanoseconds with the timebase (t * numer / denom in 64 bits, the timebase read once), stored in both.
- (void)sampleTimestamps:(MTLTimestamp *)cpuTimestamp gpuTimestamp:(MTLTimestamp *)gpuTimestamp
{
    const uint64_t ticks = mach_absolute_time();
    static const mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        return info;
    }();
    const uint64_t nanoseconds = ticks * timebase.numer / timebase.denom;
    *cpuTimestamp = nanoseconds;
    *gpuTimestamp = nanoseconds;
}

// -[AGXA10FamilyDevice disableComputeEncoderCoalescing] and -setDisableComputeEncoderCoalescing: (AGXMetalA10
// 0x21c8dc43c, 0x21c8dc450): the byte the setter stores, read back as its low bit; the device starts with 0
// (-initWithAcceleratorPort:simultaneousInstances: at 0x21c8d40a4). The driver's compute encoders read it; the plugin's
// encoders are never coalesced, so it changes nothing else.
- (BOOL)disableComputeEncoderCoalescing
{
    return [objc_getAssociatedObject(self, &kComputeCoalescingKey) unsignedCharValue] & 1;
}

- (void)setDisableComputeEncoderCoalescing:(BOOL)disable
{
    objc_setAssociatedObject(self, &kComputeCoalescingKey, @(static_cast<unsigned char>(disable)),
                             OBJC_ASSOCIATION_RETAIN);
}

@end

@implementation AGXA10XDevice (WebMetalSPI)

// -[AGXA10XDevice familyName] (AGXMetalA10 0x21c9c4598): the constant string "A10X".
- (NSString *)familyName
{
    return @"A10X";
}

@end
