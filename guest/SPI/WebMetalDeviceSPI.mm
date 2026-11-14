// The device's SPI answers (AGXA10XDevice and AGXA10FamilyDevice on 21H461).

#import "WebMetalSPI.h"
#import "../Resources/WebMetalCounterSampleBuffer.h"

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

}  // namespace

@implementation AGXA10FamilyDevice (WebMetalSPI)

// -[AGXA10FamilyDevice newCounterSampleBufferWithDescriptor:error:] (AGXMetalA10 0x21c8db494): a new
// AGXMTLCounterSampleBuffer from the descriptor (WebMetalCounterSampleBuffer.mm).
- (id<MTLCounterSampleBuffer>)newCounterSampleBufferWithDescriptor:(MTLCounterSampleBufferDescriptor *)descriptor
                                                             error:(NSError *__autoreleasing *)error
{
    return [[AGXMTLCounterSampleBuffer alloc] initWithDevice:self descriptor:descriptor error:error];
}

// -[AGXA10FamilyDevice newCounterSampleBufferWithLength:label:error:] (AGXMetalA10 0x21c8db4e0): the length form with
// options 0.
- (id)newCounterSampleBufferWithLength:(NSUInteger)length label:(NSString *)label error:(NSError *__autoreleasing *)error
{
    return [[AGXMTLCounterSampleBuffer alloc] initWithDevice:self length:length label:label options:0 error:error];
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
