// The device's SPI answers (AGXA10XDevice and AGXA10FamilyDevice on 21H461).

#import "WebMetalSPI.h"

#import <objc/runtime.h>

#include <dlfcn.h>

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

@implementation WebMetalDevice (WebMetalSPI)

// -[AGXA10FamilyDevice vendorName] (AGXMetalA10 0x21c8d4d24): the constant string "Apple".
- (NSString *)vendorName
{
    return @"Apple";
}

// -[AGXA10XDevice familyName] (AGXMetalA10 0x21c9c4598): the constant string "A10X" (AGXA10FamilyDevice's, which
// AGXA10XDevice overrides, is "A10").
- (NSString *)familyName
{
    return @"A10X";
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
